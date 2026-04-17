import json
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


@app.callback()
def _callback(
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        metavar="FILE",
        help="Path to config file (default: ~/.config/anghiari/config.toml)",
    ),
) -> None:
    from .config import load_config, set_config

    set_config(load_config(config))


def _run_llm_on_scan(query: str, scan_result: "ScanResult", top_n: int) -> "ScanResult":
    """Overlay LLM confidence + rationale onto scan results.

    Passes the scan candidates (primary + co-techniques) to the LLM reranker,
    runs subtechnique resolution, then merges confidence/rationale back onto
    the ChunkMatch objects using their technique_id as the lookup key.
    """
    from dataclasses import replace as dc_replace

    from .config import get_config
    from .mapper import _get_subtech_map, _llm_call, _llm_call_multi
    from .prompt import build_prompt, build_subtechnique_prompt
    from .scanner import ChunkMatch

    cfg = get_config()
    system = cfg.prompts.system
    max_matches = min(cfg.search.max_matches, top_n)

    # Build candidate list for the LLM — include primary + co-techniques.
    # The triggering chunk_text acts as the description proxy, giving the LLM
    # the actual evidence sentence rather than a generic technique description.
    candidates: list[dict] = []
    for m in scan_result.matches:
        candidates.append({
            "mitre_id": m.technique_id, "name": m.name,
            "tactic": m.tactic, "description": m.chunk_text,
            "score": round(m.score, 4),
        })
        for co in m.co_techniques:
            candidates.append({
                "mitre_id": co.technique_id, "name": co.name,
                "tactic": co.tactic, "description": m.chunk_text,
                "score": round(co.score, 4),
            })

    # Phase 2: LLM picks and ranks from scan candidates.
    llm_matches = _llm_call_multi(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": build_prompt(query, candidates, max_matches)},
        ],
        max_matches,
    )

    # Phase 3: subtechnique resolution (one extra LLM call per match that has subtechs).
    subtech_map = _get_subtech_map()
    resolved = []
    for match in llm_matches:
        subtechs = subtech_map.get(match.technique_id, [])
        if subtechs:
            refined = _llm_call([
                {"role": "system", "content": system},
                {"role": "user", "content": build_subtechnique_prompt(query, match, subtechs)},
            ])
            resolved.append(refined)
        else:
            resolved.append(match)

    # Build lookup: technique_id → ChunkMatch (primary or co-technique's parent).
    scan_by_id: dict[str, ChunkMatch] = {m.technique_id: m for m in scan_result.matches}
    for m in scan_result.matches:
        for co in m.co_techniques:
            scan_by_id.setdefault(co.technique_id, m)

    # Merge LLM results with scan provenance.
    merged: list[ChunkMatch] = []
    used_spans: set[tuple[int, int]] = set()  # deduplicate exact-same chunk only
    for i, tm in enumerate(resolved):
        chunk = scan_by_id.get(tm.technique_id)
        if chunk is None:
            # LLM refined to subtechnique (e.g. T1059 → T1059.001); try parent.
            chunk = scan_by_id.get(tm.technique_id.split(".")[0])
        if chunk is None:
            continue  # no provenance — skip (prevents hallucinated IDs appearing)
        span = (chunk.start, chunk.end)
        if span in used_spans:
            continue  # deduplicate only if two LLM picks resolve to the exact same chunk
        used_spans.add(span)
        # Strip any co-technique that duplicates the new primary ID.
        clean_co = [c for c in chunk.co_techniques if c.technique_id != tm.technique_id]
        merged.append(dc_replace(
            chunk,
            technique_id=tm.technique_id,
            name=tm.name,
            color_idx=i,
            confidence=tm.confidence,
            rationale=tm.rationale,
            co_techniques=clean_co,
        ))

    # Append scan matches the LLM didn't mention (cosine order, no LLM metadata).
    mentioned_parents = {m.technique_id.split(".")[0] for m in merged}
    for m in scan_result.matches:
        if m.technique_id.split(".")[0] not in mentioned_parents:
            merged.append(dc_replace(m, color_idx=len(merged)))

    return dc_replace(scan_result, matches=merged)


@app.command()
def search(
    text: Optional[str] = typer.Argument(None, help="Text to search (or use --file / stdin)"),
    file: Optional[Path] = typer.Option(None, "--file", "-f", help="Read text from a file"),
    top: int = typer.Option(8, "--top", "-n", help="Distinct technique matches to return"),
    no_llm: bool = typer.Option(
        False, "--no-llm", is_flag=True,
        help="Skip LLM reranking; use cosine scores only (faster)",
    ),
    json_output: bool = typer.Option(
        False, "--json", is_flag=True,
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
        blob = file.read_text()
    elif text:
        blob = text
    else:
        blob = sys.stdin.read()

    # Always run scan for provenance.  Fetch 2× candidates when LLM will rerank.
    scan_result = scan_text(blob, top * 2 if not no_llm else top)

    if not no_llm and scan_result.matches:
        scan_result = _run_llm_on_scan(blob, scan_result, top)

    # Trim to requested top-N after optional LLM reorder.
    from dataclasses import replace
    scan_result = replace(scan_result, matches=scan_result.matches[:top])

    if json_output:
        import dataclasses
        typer.echo(json.dumps([dataclasses.asdict(m) for m in scan_result.matches], indent=2))
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
    reload: bool = typer.Option(False, "--reload", help="Enable auto-reload (dev mode)"),
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
