from pydantic import BaseModel, Field


class TechniqueMatch(BaseModel):
    technique_id: str = Field(description="MITRE ATT&CK technique ID, e.g. T1059.001")
    name: str = Field(description="Technique name")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score between 0 and 1")
    rationale: str = Field(description="Explanation of why this technique matches the description")


class SearchResult(BaseModel):
    best_match: TechniqueMatch
    candidates: list[dict] = Field(description="Top-k raw candidates with cosine similarity scores")
