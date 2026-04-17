from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class CoTechnique:
    technique_id: str
    name: str
    tactic: str = ""
    score: float = 0.0


@dataclass(slots=True)
class TechniqueMatch:
    technique_id: str
    name: str
    tactic: str
    score: float
    chunk_text: str
    start: int
    end: int
    color_idx: int
    co_techniques: list[CoTechnique] = field(default_factory=list)


@dataclass(slots=True)
class SearchResult:
    text: str
    matches: list[TechniqueMatch]

    @property
    def best_match(self) -> TechniqueMatch | None:
        return self.matches[0] if self.matches else None


def search_result_to_dict(result: SearchResult) -> dict[str, Any]:
    payload = asdict(result)
    payload["best_match"] = asdict(result.best_match) if result.best_match else None
    return payload


def search_result_schema() -> dict[str, Any]:
    co_technique = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "technique_id": {"type": "string"},
            "name": {"type": "string"},
            "tactic": {"type": "string"},
            "score": {"type": "number"},
        },
        "required": ["technique_id", "name", "tactic", "score"],
    }
    technique_match = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "technique_id": {"type": "string"},
            "name": {"type": "string"},
            "tactic": {"type": "string"},
            "score": {"type": "number"},
            "chunk_text": {"type": "string"},
            "start": {"type": "integer"},
            "end": {"type": "integer"},
            "color_idx": {"type": "integer"},
            "co_techniques": {"type": "array", "items": co_technique},
        },
        "required": [
            "technique_id",
            "name",
            "tactic",
            "score",
            "chunk_text",
            "start",
            "end",
            "color_idx",
            "co_techniques",
        ],
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "text": {"type": "string"},
            "matches": {"type": "array", "items": technique_match},
            "best_match": {"anyOf": [technique_match, {"type": "null"}]},
        },
        "required": ["text", "matches", "best_match"],
    }
