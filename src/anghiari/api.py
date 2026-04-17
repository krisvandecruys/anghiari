"""
Litestar REST API.

Run with:  anghiari api
           uvicorn anghiari.api:app --reload     (dev)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import uvicorn
from litestar import Litestar, post
from litestar.openapi import OpenAPIConfig

from .__about__ import __version__
from .mapper import search_technique
from .models import SearchResult


@dataclass
class SearchRequest:
    query: str
    top_k: int = field(default=5)
    all_confidence: bool = field(default=False)


@post("/search", sync_to_thread=True)
def search_handler(data: SearchRequest) -> SearchResult:
    """Search for the best-matching MITRE ATT&CK technique for a free-text attack description."""
    return search_technique(data.query, data.top_k, data.all_confidence)


app = Litestar(
    route_handlers=[search_handler],
    openapi_config=OpenAPIConfig(
        title="Anghiari",
        version=__version__,
        description="Maps free-text attack descriptions to MITRE ATT&CK techniques using local ML.",
    ),
)


def run() -> None:
    from .config import get_config

    cfg = get_config().api
    uvicorn.run("anghiari.api:app", host=cfg.host, port=cfg.port, reload=False)
