from typing import Literal

from pydantic import BaseModel, Field

Confidence = Literal["GUESS", "LOW", "MEDIUM", "HIGH", "CERTAIN"]


class TechniqueMatch(BaseModel):
    technique_id: str = Field(description="MITRE ATT&CK technique ID, e.g. T1059.001")
    name: str = Field(description="Technique name")
    confidence: Confidence = Field(description="Confidence level: GUESS, LOW, MEDIUM, HIGH, CERTAIN")
    rationale: str = Field(description="Explanation of why this technique matches the description")


class SearchResult(BaseModel):
    best_match: TechniqueMatch
    candidates: list[dict] = Field(description="Top-k raw candidates with cosine similarity scores")
