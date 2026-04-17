from anghiari.prompt import build_prompt
from anghiari.scanner import multi_level_chunks


def test_build_prompt_includes_candidate_descriptions() -> None:
    prompt = build_prompt(
        "caller asked for MFA code",
        [
            {
                "mitre_id": "T1598.004",
                "name": "Phishing for Information: Spearphishing Voice",
                "tactic": "reconnaissance",
                "description": "Voice phishing used to solicit information from victims.",
                "score": 0.71,
            }
        ],
        1,
    )

    assert "Voice phishing used to solicit information from victims." in prompt
    assert "caller asked for MFA code" in prompt


def test_multi_level_chunks_extracts_sentences_and_quotes() -> None:
    text = (
        'The attacker said "send the code now". The victim complied, then reported it.'
    )

    chunks = multi_level_chunks(text)
    chunk_texts = [chunk_text for chunk_text, _, _ in chunks]

    assert any("send the code now" in chunk for chunk in chunk_texts)
    assert any("The victim complied" in chunk for chunk in chunk_texts)
