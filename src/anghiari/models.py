from typing import Literal

from pydantic import BaseModel, Field, computed_field

Confidence = Literal["GUESS", "LOW", "MEDIUM", "HIGH", "CERTAIN"]


class CoTechnique(BaseModel):
    technique_id: str = Field(description="MITRE ATT&CK technique ID")
    name: str = Field(description="Technique name")
    tactic: str = Field(default="", description="ATT&CK tactic name")
    score: float = Field(description="Cosine similarity score for this technique")


class LLMTechniqueMatch(BaseModel):
    technique_id: str = Field(description="MITRE ATT&CK technique ID, e.g. T1059.001")
    name: str = Field(description="Technique name")
    confidence: Confidence = Field(
        description="Confidence level: GUESS, LOW, MEDIUM, HIGH, CERTAIN"
    )
    rationale: str = Field(
        description="Explanation of why this technique matches the description"
    )


class LLMMatchList(BaseModel):
    matches: list[LLMTechniqueMatch]


class TechniqueMatch(BaseModel):
    technique_id: str = Field(description="MITRE ATT&CK technique ID, e.g. T1059.001")
    name: str = Field(description="Technique name")
    tactic: str = Field(default="", description="ATT&CK tactic name")
    score: float = Field(description="Cosine similarity score for the selected chunk")
    chunk_text: str = Field(description="Source text chunk that triggered the match")
    start: int = Field(description="Inclusive start offset in the original text")
    end: int = Field(description="Exclusive end offset in the original text")
    color_idx: int = Field(description="Stable colour index used by CLI rendering")
    confidence: Confidence | None = Field(
        default=None,
        description="Optional LLM confidence level for the match",
    )
    rationale: str | None = Field(
        default=None,
        description="Optional explanation of why this technique matches the description",
    )
    co_techniques: list[CoTechnique] = Field(
        default_factory=list,
        description="Other techniques that co-fired on the same chunk",
    )


class SearchResult(BaseModel):
    text: str = Field(description="Original query or report text")
    matches: list[TechniqueMatch]

    @computed_field
    @property
    def best_match(self) -> TechniqueMatch | None:
        return self.matches[0] if self.matches else None
