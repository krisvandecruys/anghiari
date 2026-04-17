import json
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import typer

if TYPE_CHECKING:
    from .scanner import ScanResult

app = typer.Typer(
    name="anghiari",
    help="Search free-text attack descriptions against MITRE ATT&CK Enterprise techniques.",
    no_args_is_help=True,
    rich_markup_mode="rich",
)

_log = logging.getLogger(__name__)


@app.callback()
def _callback(
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        metavar="FILE",
        help="Path to config file (default: ~/.config/anghiari/config.toml)",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose logging output",
    ),
) -> None:
    from .config import load_config, set_config

    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(message)s",
    )

    if verbose:
        # Silence noisy third-party loggers
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
        logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

    set_config(load_config(config))


def _run_llm_on_scan(query: str, scan_result: "ScanResult", top_n: int) -> "ScanResult":
    """Overlay LLM confidence + rationale onto scan results.

    For each chunk, passes its scan candidates (primary + co-techniques)
    to the LLM reranker to select the single best technique for that specific chunk.
    Then runs subtechnique resolution, merging confidence/rationale back onto the chunk.
    """
    from dataclasses import replace as dc_replace

    from .config import get_config
    from .mapper import _get_subtech_map, _llm_call, _llm_call_multi
    from .prompt import build_prompt, build_subtechnique_prompt
    from .scanner import ChunkMatch, CoTechnique

    cfg = get_config()
    system = cfg.prompts.system

    merged: list[ChunkMatch] = []
    subtech_map = _get_subtech_map()

    # Only process up to top_n chunks. We fetched extra in scan_text just in case,
    # but we only need to annotate the top_n chunks.
    chunks_to_process = scan_result.matches[:top_n]

    if chunks_to_process:
        _log.info(
            "Requesting LLM to evaluate %d text chunks individually...",
            len(chunks_to_process),
        )

    for i, chunk in enumerate(chunks_to_process):
        candidates = [
            {
                "mitre_id": chunk.technique_id,
                "name": chunk.name,
                "tactic": chunk.tactic,
                "description": chunk.chunk_text,
                "score": round(chunk.score, 4),
            }
        ]
        for co in chunk.co_techniques:
            candidates.append(
                {
                    "mitre_id": co.technique_id,
                    "name": co.name,
                    "tactic": co.tactic,
                    "description": chunk.chunk_text,
                    "score": round(co.score, 4),
                }
            )

        # How many slots we still need to reach top_n, ensuring every remaining chunk gets at least 1
        remaining_chunks = len(chunks_to_process) - 1 - i
        slots_needed = max(1, (top_n - len(merged)) - remaining_chunks)

        _log.info(
            "  Chunk %d/%d: %d candidates...",
            i + 1,
            len(chunks_to_process),
            len(candidates),
        )

        # LLM picks the best techniques for this specific chunk
        llm_matches = _llm_call_multi(
            [
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": build_prompt(chunk.chunk_text, candidates, slots_needed),
                },
            ],
            slots_needed,
        )

        if not llm_matches:
            # Fallback if LLM returns empty: keep the cosine primary match, but no confidence/rationale
            merged.append(dc_replace(chunk, color_idx=len(merged)))
            continue

        selected_ids = {tm.technique_id for tm in llm_matches}

        # Any candidate not selected by the LLM becomes a co-technique for the first match
        unselected = []
        for c in candidates:
            if c["mitre_id"] not in selected_ids:
                unselected.append(
                    CoTechnique(
                        technique_id=str(c["mitre_id"]),
                        name=str(c["name"]),
                        tactic=str(c["tactic"]),
                        score=float(c["score"]),
                    )
                )
        unselected.sort(key=lambda x: x.score, reverse=True)

        for j, tm in enumerate(llm_matches):
            # Remember original ID before subtechnique resolution
            orig_id = tm.technique_id

            # Subtechnique resolution
            subtechs = subtech_map.get(tm.technique_id, [])
            if subtechs:
                _log.info("  Refining %s to subtechnique...", tm.technique_id)
                tm = _llm_call(
                    [
                        {"role": "system", "content": system},
                        {
                            "role": "user",
                            "content": build_subtechnique_prompt(
                                chunk.chunk_text, tm, subtechs
                            ),
                        },
                    ]
                )

            # Recover original tactic and score for this technique if possible
            orig = next((c for c in candidates if c["mitre_id"] == orig_id), None)

            merged.append(
                dc_replace(
                    chunk,
                    technique_id=tm.technique_id,
                    name=tm.name,
                    tactic=str(orig["tactic"]) if orig else chunk.tactic,
                    score=float(orig["score"]) if orig else chunk.score,
                    color_idx=len(merged),
                    confidence=tm.confidence,
                    rationale=tm.rationale,
                    co_techniques=unselected if j == 0 else [],
                )
            )

    return dc_replace(scan_result, matches=merged)


@app.command()
def search(
    text: Optional[str] = typer.Argument(
        None, help="Text to search (or use --file / stdin)"
    ),
    file: Optional[Path] = typer.Option(
        None, "--file", "-f", help="Read text from a file"
    ),
    top: int = typer.Option(
        8, "--top", "-n", help="Distinct technique matches to return"
    ),
    no_llm: bool = typer.Option(
        False,
        "--no-llm",
        is_flag=True,
        help="Skip LLM reranking; use cosine scores only (faster)",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        is_flag=True,
        help="Output structured JSON instead of ANSI table",
    ),
) -> None:
    """[bold]Search[/bold] a threat description for MITRE ATT&CK techniques.

    Accepts any length of text — a short query phrase, a paragraph, or a full
    threat report (via --file or stdin).  Always shows which passage in the
    source triggered each match.  Pass --no-llm for fast cosine-only results;
    the default also runs the local LLM for confidence ratings and rationale.
    """
    from .scanner import render, scan_text

    if file:
        if not file.exists():
            typer.echo(f"Error: File '{file}' not found.", err=True)
            raise typer.Exit(1)
        blob = file.read_text()
    elif text:
        blob = text
    else:
        blob = sys.stdin.read()

    # Always run scan for provenance.  Fetch 2× candidates when LLM will rerank.
    top_scan = top * 2 if not no_llm else top
    _log.info("Scanning text against MITRE ATT&CK techniques (top %d)...", top_scan)
    scan_result = scan_text(blob, top_scan)

    if not no_llm and scan_result.matches:
        _log.info("Refining scan results using Nemotron LLM...")
        scan_result = _run_llm_on_scan(blob, scan_result, top)

    _log.info("Returning top %d distinct techniques.", top)

    # Trim to requested top-N after optional LLM reorder.
    from dataclasses import replace

    scan_result = replace(scan_result, matches=scan_result.matches[:top])

    if json_output:
        import dataclasses

        typer.echo(
            json.dumps([dataclasses.asdict(m) for m in scan_result.matches], indent=2)
        )
    else:
        typer.echo(render(scan_result))


@app.command()
def index(
    force: bool = typer.Option(
        False, "--force", "-f", help="Re-download STIX data even if already cached"
    ),
) -> None:
    """[bold]Build[/bold] the local vector index from MITRE ATT&CK STIX data."""
    from .indexer import main as _index

    if force:
        from .config import get_config

        cache = get_config().stix_cache
        if cache.exists():
            cache.unlink()
            typer.echo("Cleared STIX cache.")

    _index()


@app.command()
def api(
    host: str = typer.Option("0.0.0.0", "--host", help="Bind host"),
    port: int = typer.Option(8000, "--port", "-p", help="Bind port"),
    reload: bool = typer.Option(
        False, "--reload", help="Enable auto-reload (dev mode)"
    ),
) -> None:
    """[bold]Start[/bold] the Litestar REST API server."""
    import uvicorn

    uvicorn.run("anghiari.api:app", host=host, port=port, reload=reload)


@app.command()
def mcp() -> None:
    """[bold]Start[/bold] the FastMCP server (stdio transport for Claude Desktop)."""
    from .mcp import run

    run()


def main() -> None:
    app()
