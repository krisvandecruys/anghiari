import json
import logging
import sys
from pathlib import Path
from typing import Any, Optional, cast

import typer

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

    set_config(load_config(config, create_default=True))


@app.command()
def search(
    text: Optional[str] = typer.Argument(
        None, help="Text to search (or use --file / stdin)"
    ),
    file: Optional[Path] = typer.Option(
        None, "--file", "-f", help="Read text from a file"
    ),
    top: int = typer.Option(
        5, "--top", "-n", help="Distinct technique matches to return"
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
    all_confidence: bool = typer.Option(
        False,
        "--all-confidence",
        is_flag=True,
        help="Include lower confidence matches (GUESS, LOW, MEDIUM)",
    ),
) -> None:
    """[bold]Search[/bold] a threat description for MITRE ATT&CK techniques.

    Accepts any length of text — a short query phrase, a paragraph, or a full
    threat report (via --file or stdin).  Always shows which passage in the
    source triggered each match.  Pass --no-llm for fast cosine-only results.
    JSON output matches the API and MCP response shape.
    """
    from .mapper import search_technique
    from .models import CoTechnique as ResultCoTechnique
    from .models import SearchResult, TechniqueMatch
    from .scanner import ChunkMatch, CoTechnique, ScanResult, render

    if file:
        if not file.exists():
            typer.echo(f"Error: File '{file}' not found.", err=True)
            raise typer.Exit(1)
        blob = file.read_text()
    elif text:
        blob = text
    else:
        blob = sys.stdin.read()

    from rich.console import Console

    console = Console(stderr=True)
    with console.status("Starting search...", spinner="dots") as status:
        if no_llm:
            from .scanner import scan_text

            status.update(
                f"Scanning text against MITRE ATT&CK techniques (top {top})..."
            )
            _log.info("Scanning text against MITRE ATT&CK techniques (top %d)...", top)
            scan_result = scan_text(blob, top)
            scan_result = ScanResult(
                text=scan_result.text,
                matches=scan_result.matches[:top],
            )
            result = SearchResult(
                text=scan_result.text,
                matches=[
                    TechniqueMatch(
                        technique_id=m.technique_id,
                        name=m.name,
                        tactic=m.tactic,
                        score=m.score,
                        chunk_text=m.chunk_text,
                        start=m.start,
                        end=m.end,
                        color_idx=m.color_idx,
                        confidence=cast(Any, m.confidence),
                        rationale=m.rationale,
                        co_techniques=[
                            ResultCoTechnique(
                                technique_id=co.technique_id,
                                name=co.name,
                                tactic=co.tactic,
                                score=co.score,
                            )
                            for co in m.co_techniques
                        ],
                    )
                    for m in scan_result.matches
                ],
            ).model_dump()
        else:
            status.update("Running shared search pipeline...")
            _log.info("Running shared search pipeline...")
            search_result = search_technique(blob, top, all_confidence)
            result = search_result.model_dump()
            scan_result = ScanResult(
                text=search_result.text,
                matches=[
                    ChunkMatch(
                        technique_id=m.technique_id,
                        name=m.name,
                        description="",
                        tactic=m.tactic,
                        score=m.score,
                        chunk_text=m.chunk_text,
                        start=m.start,
                        end=m.end,
                        color_idx=m.color_idx,
                        confidence=m.confidence,
                        rationale=m.rationale,
                        co_techniques=[
                            CoTechnique(
                                technique_id=co.technique_id,
                                name=co.name,
                                description="",
                                tactic=co.tactic,
                                score=co.score,
                            )
                            for co in m.co_techniques
                        ],
                    )
                    for m in search_result.matches
                ],
            )

    if json_output:
        typer.echo(json.dumps(result, indent=2))
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
