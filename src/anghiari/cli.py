import json
from pathlib import Path

import typer

app = typer.Typer(
    name="anghiari",
    help="Search free-text attack descriptions against MITRE ATT&CK Enterprise techniques.",
    no_args_is_help=True,
    rich_markup_mode="rich",
)


@app.command()
def search(
    query: str = typer.Argument(..., help="Free-text description of the attack behavior"),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Candidates to retrieve before LLM reranking"),
    pretty: bool = typer.Option(True, "--pretty/--no-pretty", help="Pretty-print JSON output"),
) -> None:
    """[bold]Search[/bold] for the best-matching MITRE ATT&CK technique."""
    from .mapper import search_technique

    result = search_technique(query, top_k)
    indent = 2 if pretty else None
    typer.echo(json.dumps(result.model_dump(), indent=indent))


@app.command()
def index(
    force: bool = typer.Option(
        False, "--force", "-f", help="Re-download STIX data even if already cached"
    ),
) -> None:
    """[bold]Build[/bold] the local vector index from MITRE ATT&CK STIX data."""
    from .indexer import main as _index

    if force:
        cache = Path("data/enterprise-attack.json")
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
