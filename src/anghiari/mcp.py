"""
FastMCP server.

Run with:  anghiari mcp                                          (stdio, for Claude Desktop)
           fastmcp run anghiari.mcp:mcp --transport streamable-http --port 8001   (HTTP)
"""

import sys

from fastmcp import FastMCP

from .mapper import search_technique, validate_top_k, warmup_backend
from .models import search_result_to_dict


def _mcp_debug(message: str) -> None:
    # Keep protocol traffic on stdout; emit diagnostics on stderr only.
    print(f"[anghiari-mcp] {message}", file=sys.stderr, flush=True)

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
    resolved_top_k = validate_top_k(top_k)
    preview = query.strip().replace("\n", " ")[:120]
    _mcp_debug(
        f"json request start top_k={resolved_top_k} all_confidence={all_confidence} query={preview!r}"
    )
    result = search_technique(query, resolved_top_k, all_confidence)
    _mcp_debug(
        f"json request done matches={len(result.matches)} best={result.best_match.technique_id if result.best_match else 'none'}"
    )
    return search_result_to_dict(result)


@mcp.tool()
def search_attack_technique_best(
    query: str, top_k: int | None = None, all_confidence: bool = False
) -> str:
    """Return a concise multi-line best answer for an LLM client."""
    resolved_top_k = validate_top_k(top_k)
    preview = query.strip().replace("\n", " ")[:120]
    _mcp_debug(
        f"best request start top_k={resolved_top_k} all_confidence={all_confidence} query={preview!r}"
    )
    result = search_technique(query, resolved_top_k, all_confidence)
    _mcp_debug(
        f"best request done matches={len(result.matches)} best={result.best_match.technique_id if result.best_match else 'none'}"
    )
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
    _mcp_debug("server starting")
    _mcp_debug("warming backend")
    warmup_backend()
    _mcp_debug("backend warmup complete")
    mcp.run()
