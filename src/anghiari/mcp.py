"""
FastMCP server.

Run with:  anghiari mcp                                          (stdio, for Claude Desktop)
           fastmcp run anghiari.mcp:mcp --transport streamable-http --port 8001   (HTTP)
"""

from fastmcp import FastMCP

from .mapper import search_technique

mcp = FastMCP(
    name="Anghiari",
    instructions=(
        "Searches for and maps free-text descriptions of attack behaviors to MITRE ATT&CK "
        "Enterprise techniques. Uses local embeddings (Harrier) for semantic search and a "
        "local LLM (Nemotron) for reasoning. Returns JSON containing the original text, "
        "chunk-grounded matches, and the best match when available."
    ),
)


@mcp.tool()
def search_attack_technique(
    query: str, top_k: int | None = None, all_confidence: bool = False
) -> dict:
    """Search for the best-matching MITRE ATT&CK technique for a free-text attack description.

    Args:
        query: Natural-language description of the observed attack behavior.
        top_k: Maximum number of technique matches to return.
               Defaults to the value configured in config.toml (default: 5).
        all_confidence: If true, include matches with confidence below HIGH.

    Returns:
        A JSON-compatible dict matching the REST API response shape:
          - text: original query text
          - matches: chunk-grounded technique matches
          - best_match: first entry in matches, when present
    """
    return search_technique(query, top_k, all_confidence).model_dump()


def run() -> None:
    mcp.run()
