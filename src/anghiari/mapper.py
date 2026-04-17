"""
Core mapping logic: scanner retrieval plus dedicated reranking.
"""

import contextlib
import dataclasses
import json
import logging
import os
from typing import Any

import chromadb
from sentence_transformers import CrossEncoder

from .models import CoTechnique, SearchResult, TechniqueMatch
from .scanner import ScanResult, scan_text

_log = logging.getLogger(__name__)

COLLECTION_NAME = "mitre_techniques"

_collection = None
_reranker: CrossEncoder | None = None
_subtech_map: dict[str, list[dict]] | None = None


@contextlib.contextmanager
def _quiet_stdouterr():
    saved_stdout = os.dup(1)
    saved_stderr = os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)
    os.dup2(devnull, 2)
    os.close(devnull)
    try:
        yield
    finally:
        os.dup2(saved_stdout, 1)
        os.dup2(saved_stderr, 2)
        os.close(saved_stdout)
        os.close(saved_stderr)


def _get_collection():
    global _collection
    if _collection is None:
        from .config import get_config

        chroma_dir = get_config().chroma_dir
        if not chroma_dir.exists():
            raise RuntimeError(
                f"ChromaDB index not found at {chroma_dir}. Run `anghiari index` first."
            )
        client = chromadb.PersistentClient(path=str(chroma_dir))
        _collection = client.get_collection(COLLECTION_NAME)
    return _collection


def _get_subtech_map() -> dict[str, list[dict]]:
    global _subtech_map
    if _subtech_map is None:
        from .config import get_config

        subtech_map_file = get_config().subtech_map
        if not subtech_map_file.exists():
            raise RuntimeError(
                f"Subtechnique map not found at {subtech_map_file}. "
                "Run `anghiari index` first."
            )
        _subtech_map = json.loads(subtech_map_file.read_text())
    return _subtech_map


def _get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        from .config import get_config

        cfg = get_config().reranker
        _log.info("Loading reranker (%s)", cfg.model_id)
        with _quiet_stdouterr():
            _reranker = CrossEncoder(
                cfg.model_id,
                prompts={"attack_match": cfg.instruction},
            )
    return _reranker


def _to_result_match(
    chunk: Any, *, co_techniques: list[CoTechnique] | None = None
) -> TechniqueMatch:
    return TechniqueMatch(
        technique_id=chunk.technique_id,
        name=chunk.name,
        tactic=chunk.tactic,
        score=chunk.score,
        chunk_text=chunk.chunk_text,
        start=chunk.start,
        end=chunk.end,
        color_idx=chunk.color_idx,
        co_techniques=(
            co_techniques
            if co_techniques is not None
            else [
                CoTechnique(
                    technique_id=co.technique_id,
                    name=co.name,
                    tactic=co.tactic,
                    score=co.score,
                )
                for co in chunk.co_techniques
            ]
        ),
    )


def _apply_subtechnique_upgrade(chunk: Any) -> Any:
    subtech_map = _get_subtech_map()
    subtechs = subtech_map.get(chunk.technique_id, [])
    if not subtechs:
        return chunk

    chunk_lower = chunk.chunk_text.lower()
    for sub in subtechs:
        sub_name = str(sub.get("name", ""))
        description = str(sub.get("description", ""))
        if sub_name and sub_name.lower() in chunk_lower:
            return dataclasses.replace(
                chunk,
                technique_id=sub["mitre_id"],
                name=sub["name"],
                description=description,
            )
        if description and any(
            token in chunk_lower for token in description.lower().split()[:4]
        ):
            return dataclasses.replace(
                chunk,
                technique_id=sub["mitre_id"],
                name=sub["name"],
                description=description,
            )
    return chunk


def _rerank_matches(matches: list[Any], top_n: int) -> list[TechniqueMatch]:
    reranked: list[TechniqueMatch] = []
    reranker = _get_reranker()

    for chunk in matches[:top_n]:
        candidates = [chunk] + list(chunk.co_techniques)
        docs = []
        ids = []
        names = {}
        tactics = {}
        scores = {}
        descriptions = {}

        for cand in candidates:
            description = getattr(cand, "description", "")
            docs.append(
                f"[{cand.technique_id}] {cand.name}\n"
                f"Tactic: {cand.tactic or 'unknown'}\n"
                f"Description: {description}\n"
                f"Chunk: {chunk.chunk_text}"
            )
            ids.append(cand.technique_id)
            names[cand.technique_id] = cand.name
            tactics[cand.technique_id] = cand.tactic
            scores[cand.technique_id] = cand.score
            descriptions[cand.technique_id] = description

        rankings = reranker.rank(chunk.chunk_text, docs, prompt_name="attack_match")
        if not rankings:
            reranked.append(_to_result_match(chunk))
            reranked[-1].color_idx = len(reranked) - 1
            continue

        best = rankings[0]
        chosen_id = ids[best["corpus_id"]]
        selected_ids = {ids[item["corpus_id"]] for item in rankings[1:]}
        selected_chunk = dataclasses.replace(
            chunk,
            technique_id=chosen_id,
            name=names[chosen_id],
            tactic=tactics[chosen_id],
            description=descriptions[chosen_id],
            score=float(best["score"]),
            color_idx=len(reranked),
        )
        selected_chunk = _apply_subtechnique_upgrade(selected_chunk)
        reranked.append(
            _to_result_match(
                selected_chunk,
                co_techniques=[
                    CoTechnique(
                        technique_id=cand.technique_id,
                        name=cand.name,
                        tactic=cand.tactic,
                        score=scores[cand.technique_id],
                    )
                    for cand in candidates
                    if cand.technique_id != chosen_id
                    and cand.technique_id in selected_ids
                ],
            )
        )

    return reranked


def _passes_default_threshold(score: float) -> bool:
    from .config import get_config

    return score >= get_config().reranker.high_threshold


def validate_top_k(top_k: int | None) -> int:
    from .config import get_config

    if top_k is None:
        top_k = get_config().search.top_k
    if not 1 <= top_k <= 5:
        raise ValueError("top_k must be between 1 and 5")
    return top_k


def search_technique(
    query: str, top_k: int | None = None, all_confidence: bool = False
) -> SearchResult:
    """Map a free-text attack description to one or more MITRE ATT&CK techniques."""
    top_k = validate_top_k(top_k)

    top_scan = top_k * 2
    _log.info("Scanning text against MITRE ATT&CK techniques (top %d)...", top_scan)
    scan_result: ScanResult = scan_text(query, top_scan)
    if not scan_result.matches:
        return SearchResult(text=query, matches=[])

    _log.info("Running Qwen3-Reranker-4B over chunk-level candidates...")
    resolved = _rerank_matches(scan_result.matches, top_k)

    if not all_confidence:
        _log.info(
            "Filtering matches to the configured high-confidence score threshold..."
        )
        resolved = [m for m in resolved if _passes_default_threshold(m.score)]
        for idx, match in enumerate(resolved):
            resolved[idx].color_idx = idx

    return SearchResult(text=query, matches=resolved[:top_k])
