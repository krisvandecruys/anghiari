from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .models import LLMTechniqueMatch, TechniqueMatch

_JSON_SCHEMA_SINGLE = (
    '{"technique_id": "T...", "name": "...", "confidence": "HIGH", "rationale": "..."}\n'
    "confidence must be exactly one of (ordered lowest to highest): GUESS < LOW < MEDIUM < HIGH < CERTAIN\n"
    "  GUESS = weak signal, speculative match;  CERTAIN = definitively matches, near-unmistakable"
)

_JSON_SCHEMA_MULTI_TMPL = (
    '{{"matches": [{{"technique_id": "T...", "name": "...", "confidence": "HIGH", "rationale": "..."}}, ...]}}\n'
    "Return up to {max_matches} entries. If asked for multiple, evaluate the top candidates even if one dominates, assigning lower confidence (e.g. GUESS or LOW) to the weaker ones.\n"
    "confidence must be exactly one of (ordered lowest to highest): GUESS < LOW < MEDIUM < HIGH < CERTAIN\n"
    "  GUESS = weak signal, speculative match;  CERTAIN = definitively matches, near-unmistakable"
)


def build_prompt(query: str, candidates: list[dict], max_matches: int = 1) -> str:
    from .config import get_config

    trunc = get_config().prompts.description_truncate_phase1
    candidate_block = "\n".join(
        f"{i + 1}. [{c['mitre_id']}] {c['name']} (tactic: {c.get('tactic', 'unknown')})\n"
        f"   {c['description'][:trunc]}"
        for i, c in enumerate(candidates)
    )
    schema = _JSON_SCHEMA_MULTI_TMPL.format(max_matches=max_matches)
    return (
        f"Attack description: {query}\n\n"
        f"Candidate techniques:\n{candidate_block}\n\n"
        f"Return JSON with this exact schema:\n{schema}"
    )


def build_subtechnique_prompt(
    query: str,
    parent: "TechniqueMatch | LLMTechniqueMatch",
    subtechs: list[dict],
) -> str:
    from .config import get_config

    trunc = get_config().prompts.description_truncate_phase2
    subtech_block = "\n".join(
        f"{i + 1}. [{s['mitre_id']}] {s['name']}\n   {s['description'][:trunc]}"
        for i, s in enumerate(subtechs)
    )
    return (
        f"Attack description: {query}\n\n"
        f"You identified [{parent.technique_id}] {parent.name} as the best parent technique.\n"
        f"Now pick the most specific subtechnique below, or return the parent ID "
        f"if none of them fit better:\n\n"
        f"{subtech_block}\n\n"
        f"Return JSON with this exact schema:\n{_JSON_SCHEMA_SINGLE}"
    )
