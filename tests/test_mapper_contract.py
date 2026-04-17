import json

from typer.testing import CliRunner

from anghiari.cli import app
from anghiari.mapper import search_technique, validate_top_k
from anghiari.mcp import search_attack_technique_best, search_attack_technique_json
from anghiari.models import (
    CoTechnique as ResultCoTechnique,
    SearchResult,
    TechniqueMatch,
    search_result_to_dict,
)
from anghiari.scanner import ChunkMatch, CoTechnique, ScanResult


runner = CliRunner()


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


def _result_matches() -> list[TechniqueMatch]:
    return [
        TechniqueMatch(
            technique_id="T1110",
            name="Brute Force",
            tactic="credential-access",
            score=0.91,
            chunk_text="The attacker repeatedly guessed passwords.",
            start=0,
            end=42,
            color_idx=0,
            co_techniques=[
                ResultCoTechnique(
                    technique_id="T1078",
                    name="Valid Accounts",
                    tactic="defense-evasion",
                    score=0.88,
                )
            ],
        )
    ]


def _stub_reranker(score: float = 0.91, corpus_id: int = 0):
    return type(
        "StubReranker",
        (),
        {
            "rank": lambda self, query, docs, **kwargs: [
                {"corpus_id": corpus_id, "score": score}
            ]
        },
    )()


def test_search_technique_returns_shared_contract(monkeypatch) -> None:
    monkeypatch.setattr(
        "anghiari.mapper.scan_text", lambda query, top_n: _dummy_scan_result()
    )
    monkeypatch.setattr("anghiari.mapper._get_subtech_map", lambda: {})
    monkeypatch.setattr("anghiari.mapper._get_reranker", lambda: _stub_reranker())

    result = search_technique("The attacker repeatedly guessed passwords.", top_k=1)
    payload = search_result_to_dict(result)

    assert payload["text"] == "The attacker repeatedly guessed passwords."
    assert payload["best_match"]["technique_id"] == "T1110"
    assert payload["matches"][0]["co_techniques"] == []
    assert "confidence" not in payload["matches"][0]
    assert "rationale" not in payload["matches"][0]


def test_search_technique_filters_low_score(monkeypatch) -> None:
    monkeypatch.setattr(
        "anghiari.mapper.scan_text", lambda query, top_n: _dummy_scan_result()
    )
    monkeypatch.setattr("anghiari.mapper._get_subtech_map", lambda: {})
    monkeypatch.setattr(
        "anghiari.mapper._get_reranker", lambda: _stub_reranker(score=0.10)
    )

    result = search_technique("The attacker repeatedly guessed passwords.", top_k=1)

    assert result.matches == []
    assert result.best_match is None


def test_validate_top_k_guardrails() -> None:
    assert validate_top_k(1) == 1
    assert validate_top_k(5) == 5

    for invalid in (0, -1, 6):
        try:
            validate_top_k(invalid)
        except ValueError as exc:
            assert str(exc) == "top_k must be between 1 and 5"
        else:
            raise AssertionError("expected ValueError")


def test_cli_json_matches_programmatic_contract(monkeypatch, tmp_path) -> None:
    result = SearchResult(
        text="The attacker repeatedly guessed passwords.",
        matches=_result_matches(),
    )
    input_file = tmp_path / "input.txt"
    input_file.write_text(result.text)

    monkeypatch.setattr(
        "anghiari.mapper.search_technique", lambda text, top_k, all_confidence: result
    )

    cli_result = runner.invoke(app, ["search", "--json", "--file", str(input_file)])

    assert cli_result.exit_code == 0
    assert json.loads(cli_result.stdout) == search_result_to_dict(result)


def test_cli_no_reranking_keeps_score_only_schema(monkeypatch, tmp_path) -> None:
    input_file = tmp_path / "input.txt"
    input_file.write_text("The attacker repeatedly guessed passwords.")
    monkeypatch.setattr(
        "anghiari.scanner.scan_text", lambda text, top_n: _dummy_scan_result()
    )

    cli_result = runner.invoke(
        app, ["search", "--json", "--no-reranking", "--file", str(input_file)]
    )

    assert cli_result.exit_code == 0
    payload = json.loads(cli_result.stdout)
    assert "confidence" not in payload["matches"][0]
    assert "rationale" not in payload["matches"][0]


def test_mcp_json_and_text_tools(monkeypatch) -> None:
    result = SearchResult(
        text="The attacker repeatedly guessed passwords.",
        matches=_result_matches(),
    )
    monkeypatch.setattr(
        "anghiari.mcp.search_technique", lambda query, top_k, all_confidence: result
    )

    assert search_attack_technique_json("x", top_k=1) == search_result_to_dict(result)
    text = search_attack_technique_best("x", top_k=1)
    assert "T1110 Brute Force" in text
    assert "Score: 0.910" in text
