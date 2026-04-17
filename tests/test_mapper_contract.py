from anghiari.mapper import search_technique
from anghiari.scanner import ChunkMatch, CoTechnique, ScanResult


def _dummy_scan_result() -> ScanResult:
    return ScanResult(
        text="The attacker repeatedly guessed passwords.",
        matches=[
            ChunkMatch(
                technique_id="T1110",
                name="Brute Force",
                description="Guess passwords repeatedly.",
                tactic="credential-access",
                score=0.91,
                chunk_text="The attacker repeatedly guessed passwords.",
                start=0,
                end=42,
                color_idx=0,
                co_techniques=[
                    CoTechnique(
                        technique_id="T1078",
                        name="Valid Accounts",
                        description="Use legitimate credentials.",
                        tactic="defense-evasion",
                        score=0.88,
                    )
                ],
            )
        ],
    )


def test_search_technique_returns_shared_contract(monkeypatch) -> None:
    monkeypatch.setattr(
        "anghiari.mapper.scan_text", lambda query, top_n: _dummy_scan_result()
    )
    monkeypatch.setattr(
        "anghiari.mapper._llm_call_multi",
        lambda messages, max_matches: [
            type(
                "LLMMatch",
                (),
                {
                    "technique_id": "T1110",
                    "name": "Brute Force",
                    "confidence": "HIGH",
                    "rationale": "Repeated password guessing matches brute force.",
                },
            )()
        ],
    )
    monkeypatch.setattr("anghiari.mapper._get_subtech_map", lambda: {})

    result = search_technique("The attacker repeatedly guessed passwords.", top_k=1)
    payload = result.model_dump()

    assert payload["text"] == "The attacker repeatedly guessed passwords."
    assert payload["best_match"]["technique_id"] == "T1110"
    assert (
        payload["matches"][0]["chunk_text"]
        == "The attacker repeatedly guessed passwords."
    )
    assert payload["matches"][0]["co_techniques"][0]["technique_id"] == "T1078"


def test_search_technique_filters_low_confidence(monkeypatch) -> None:
    monkeypatch.setattr(
        "anghiari.mapper.scan_text", lambda query, top_n: _dummy_scan_result()
    )
    monkeypatch.setattr(
        "anghiari.mapper._llm_call_multi",
        lambda messages, max_matches: [
            type(
                "LLMMatch",
                (),
                {
                    "technique_id": "T1110",
                    "name": "Brute Force",
                    "confidence": "LOW",
                    "rationale": "Weak match.",
                },
            )()
        ],
    )
    monkeypatch.setattr("anghiari.mapper._get_subtech_map", lambda: {})

    result = search_technique("The attacker repeatedly guessed passwords.", top_k=1)

    assert result.matches == []
    assert result.best_match is None
