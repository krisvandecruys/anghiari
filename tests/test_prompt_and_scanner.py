from anghiari.scanner import _is_parent_sub_pair, _is_subtechnique, multi_level_chunks


def test_multi_level_chunks_extracts_sentences_and_quotes() -> None:
    text = (
        'The attacker said "send the code now". The victim complied, then reported it.'
    )

    chunks = multi_level_chunks(text)
    chunk_texts = [chunk_text for chunk_text, _, _ in chunks]

    assert any("send the code now" in chunk for chunk in chunk_texts)
    assert any("The victim complied" in chunk for chunk in chunk_texts)


def test_parent_subtechnique_helpers() -> None:
    assert _is_parent_sub_pair("T1059", "T1059.001")
    assert _is_subtechnique("T1059.001")
    assert not _is_subtechnique("T1059")
