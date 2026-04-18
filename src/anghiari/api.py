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
from litestar.exceptions import ValidationException

from . import __version__
from .mapper import search_technique, validate_top_k
from .models import search_result_to_dict


@dataclass
class SearchRequest:
    query: str
    top_k: int = field(default=5)
    all_confidence: bool = field(default=False)


@post("/search", sync_to_thread=True)
def search_handler(data: SearchRequest) -> dict:
    """Search for the best-matching MITRE ATT&CK technique for a free-text attack description."""
    try:
        top_k = validate_top_k(data.top_k)
    except ValueError as exc:
        raise ValidationException(str(exc)) from exc
    return search_result_to_dict(
        search_technique(data.query, top_k, data.all_confidence)
    )


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
