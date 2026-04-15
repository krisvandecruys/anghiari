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

from .mapper import search_technique
from .models import SearchResult


@dataclass
class SearchRequest:
    query: str
    top_k: int = field(default=5)


@post("/search", sync_to_thread=True)
def search_handler(data: SearchRequest) -> SearchResult:
    """Search for the best-matching MITRE ATT&CK technique for a free-text attack description."""
    return search_technique(data.query, data.top_k)


app = Litestar(
    route_handlers=[search_handler],
    openapi_config=OpenAPIConfig(
        title="Anghiari",
        version="0.1.0",
        description="Maps free-text attack descriptions to MITRE ATT&CK techniques using local ML.",
    ),
)


def run() -> None:
    uvicorn.run("anghiari.api:app", host="0.0.0.0", port=8000, reload=False)
