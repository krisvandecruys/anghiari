"""
FastMCP server.

Run with:  anghiari mcp                                          (stdio, for Claude Desktop)
           fastmcp run anghiari.mcp:mcp --transport streamable-http --port 8001   (HTTP)
"""

from fastmcp import FastMCP

from .mapper import search_technique, validate_top_k
from .models import search_result_to_dict

mcp = FastMCP(
    name="Anghiari",
    instructions=(
        "Searches for and maps free-text descriptions of attack behaviors to MITRE ATT&CK "
        "Enterprise techniques. Uses local embeddings (Harrier) for semantic search and "
        "Qwen3-Reranker-4B for local reranking. Returns JSON containing the original text, "
        "chunk-grounded matches, and the best match when available."
    ),
)


@mcp.tool()
def search_attack_technique_json(
    query: str, top_k: int | None = None, all_confidence: bool = False
) -> dict:
    """Search for the best-matching MITRE ATT&CK technique for a free-text attack description.

    Args:
        query: Natural-language description of the observed attack behavior.
        top_k: Maximum number of technique matches to return.
               Defaults to the value configured in config.toml (default: 5).
        all_confidence: If true, include matches below the high-confidence score threshold.

    Returns:
        A JSON-compatible dict matching the REST API response shape:
          - text: original query text
          - matches: chunk-grounded technique matches
          - best_match: first entry in matches, when present
    """
    return search_result_to_dict(
        search_technique(query, validate_top_k(top_k), all_confidence)
    )


@mcp.tool()
def search_attack_technique_best(
    query: str, top_k: int | None = None, all_confidence: bool = False
) -> str:
    """Return a concise multi-line best answer for an LLM client."""
    result = search_technique(query, validate_top_k(top_k), all_confidence)
    if not result.best_match:
        return "No matching technique found."

    best = result.best_match
    return (
        f"{best.technique_id} {best.name}\n"
        f"Tactic: {best.tactic or 'unknown'}\n"
        f"Score: {best.score:.3f}\n"
        f"Source: {best.chunk_text}"
    )


def run() -> None:
    mcp.run()
