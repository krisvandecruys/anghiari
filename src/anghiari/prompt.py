from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .models import TechniqueMatch

SYSTEM = (
    "You are a MITRE ATT&CK expert. Given an attack description and candidate techniques, "
    "identify the single best match. Respond ONLY with valid JSON — no markdown, no extra text."
)

_JSON_SCHEMA = '{"technique_id": "T...", "name": "...", "confidence": 0.0, "rationale": "..."}'


def build_prompt(query: str, candidates: list[dict]) -> str:
    candidate_block = "\n".join(
        f"{i + 1}. [{c['mitre_id']}] {c['name']} (tactic: {c.get('tactic', 'unknown')})\n"
        f"   {c['description'][:300]}"
        for i, c in enumerate(candidates)
    )
    return (
        f"Attack description: {query}\n\n"
        f"Candidate techniques:\n{candidate_block}\n\n"
        f"Return JSON with this exact schema:\n{_JSON_SCHEMA}"
    )


def build_subtechnique_prompt(
    query: str,
    parent: "TechniqueMatch",
    subtechs: list[dict],
) -> str:
    subtech_block = "\n".join(
        f"{i + 1}. [{s['mitre_id']}] {s['name']}\n   {s['description'][:400]}"
        for i, s in enumerate(subtechs)
    )
    return (
        f"Attack description: {query}\n\n"
        f"You identified [{parent.technique_id}] {parent.name} as the best parent technique.\n"
        f"Now pick the most specific subtechnique below, or return the parent ID "
        f"if none of them fit better:\n\n"
        f"{subtech_block}\n\n"
        f"Return JSON with this exact schema:\n{_JSON_SCHEMA}"
    )
