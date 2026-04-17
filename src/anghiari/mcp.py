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
        "local LLM (Nemotron) for reasoning. Returns technique ID, name, confidence, "
        "rationale, and top-k candidates."
    ),
)


@mcp.tool()
def search_attack_technique(
    query: str, top_k: int | None = None, all_confidence: bool = False
) -> dict:
    """Search for the best-matching MITRE ATT&CK technique for a free-text attack description.

    Args:
        query: Natural-language description of the observed attack behavior.
        top_k: Number of candidate techniques to retrieve before LLM reranking.
               Defaults to the value configured in config.toml (default: 5).
        all_confidence: If true, include matches with confidence below HIGH.

    Returns:
        A dict with:
          - best_match: { technique_id, name, confidence, rationale }
          - candidates: top-k raw results with cosine similarity scores
    """
    return search_technique(query, top_k, all_confidence).model_dump()


def run() -> None:
    mcp.run()
