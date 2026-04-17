from anghiari.scanner import _span_overlaps, ChunkMatch, CoTechnique


def test_span_overlap_thresholds() -> None:
    assert _span_overlaps(0, 10, 5, 12)
    assert not _span_overlaps(0, 10, 11, 20)


def test_chunkmatch_can_hold_multiple_cofires() -> None:
    match = ChunkMatch(
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
            ),
            CoTechnique(
                technique_id="T1059",
                name="Command and Scripting Interpreter",
                description="Run commands through interpreters.",
                tactic="execution",
                score=0.86,
            ),
        ],
    )

    assert len(match.co_techniques) == 2
    assert match.co_techniques[1].technique_id == "T1059"
