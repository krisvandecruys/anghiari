"""
Core mapping logic: embed → vector search → LLM phase 1 → LLM phase 2 (subtechnique).

Both models are downloaded automatically from HuggingFace Hub on first use
and cached in ~/.cache/huggingface/hub/.
"""

import contextlib
import dataclasses
import json
import logging
import os
import warnings
from typing import Any, cast

import chromadb
from llama_cpp import Llama

from .models import (
    CoTechnique,
    LLMMatchList,
    LLMTechniqueMatch,
    SearchResult,
    TechniqueMatch,
    llm_match_list_schema,
    llm_technique_match_schema,
)
from .prompt import build_prompt, build_subtechnique_prompt
from .scanner import ScanResult, scan_text

# llama-cpp-python passes a deprecated kwarg to hf_hub_download — not our bug, filter it.
warnings.filterwarnings(
    "ignore",
    message="The `local_dir_use_symlinks` argument is deprecated",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=".*unauthenticated requests.*",
)

_log = logging.getLogger(__name__)


@contextlib.contextmanager
def _quiet_stderr():
    """Redirect the C-level stderr fd to /dev/null.

    llama.cpp prints directly to fd 2 (bypassing Python logging), so the only
    reliable way to suppress those messages is at the OS level.
    """
    devnull = os.open(os.devnull, os.O_WRONLY)
    saved = os.dup(2)
    os.dup2(devnull, 2)
    os.close(devnull)
    try:
        yield
    finally:
        os.dup2(saved, 2)
        os.close(saved)


COLLECTION_NAME = "mitre_techniques"

# ── Lazy singletons ───────────────────────────────────────────────────────────
_collection = None
_llm: Llama | None = None
_subtech_map: dict[str, list[dict]] | None = None


class LLMStructuredOutputError(RuntimeError):
    pass


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


def _get_llm() -> Llama:
    global _llm
    if _llm is None:
        from .config import get_config

        llm_cfg = get_config().llm
        _log.info("Loading LLM (%s / %s)", llm_cfg.repo_id, llm_cfg.filename)
        with _quiet_stderr():
            _llm = Llama.from_pretrained(
                repo_id=llm_cfg.repo_id,
                filename=llm_cfg.filename,
                n_ctx=llm_cfg.n_ctx,
                n_gpu_layers=llm_cfg.n_gpu_layers,
                verbose=False,
                chat_format=llm_cfg.chat_format,
            )
    return _llm


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


def _parse_llm_technique_match(raw: str) -> LLMTechniqueMatch:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise LLMStructuredOutputError("LLM returned invalid JSON") from exc

    if not isinstance(payload, dict):
        raise LLMStructuredOutputError("LLM returned non-object JSON")

    required = {"technique_id", "name", "confidence", "rationale"}
    if set(payload) != required:
        raise LLMStructuredOutputError("LLM returned unexpected fields")

    confidence = payload["confidence"]
    if confidence not in {"GUESS", "LOW", "MEDIUM", "HIGH", "CERTAIN", "UNKNOWN"}:
        raise LLMStructuredOutputError("LLM returned invalid confidence value")

    if not all(isinstance(payload[key], str) for key in required):
        raise LLMStructuredOutputError("LLM returned non-string technique fields")

    return LLMTechniqueMatch(
        technique_id=payload["technique_id"],
        name=payload["name"],
        confidence=payload["confidence"],
        rationale=payload["rationale"],
    )


def _parse_llm_match_list(raw: str) -> LLMMatchList:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise LLMStructuredOutputError("LLM returned invalid JSON") from exc

    if not isinstance(payload, dict) or set(payload) != {"matches"}:
        raise LLMStructuredOutputError("LLM returned invalid match-list envelope")

    matches = payload["matches"]
    if not isinstance(matches, list):
        raise LLMStructuredOutputError("LLM returned non-list matches")

    return LLMMatchList(
        matches=[_parse_llm_technique_match(json.dumps(m)) for m in matches]
    )


def _llm_call(messages: list[dict[str, str]]) -> LLMTechniqueMatch:
    from .config import get_config

    llm_cfg = get_config().llm
    response = cast(
        Any,
        _get_llm().create_chat_completion(
            messages=cast(Any, messages),
            response_format={
                "type": "json_object",
                "schema": llm_technique_match_schema(),
            },
            max_tokens=llm_cfg.max_tokens,
            temperature=llm_cfg.temperature,
        ),
    )
    raw = cast(str | None, response["choices"][0]["message"]["content"])
    if not raw:
        raise LLMStructuredOutputError("LLM returned empty content")
    return _parse_llm_technique_match(raw)


def _llm_call_multi(
    messages: list[dict[str, str]], max_matches: int
) -> list[LLMTechniqueMatch]:
    from .config import get_config

    llm_cfg = get_config().llm
    response = cast(
        Any,
        _get_llm().create_chat_completion(
            messages=cast(Any, messages),
            response_format={
                "type": "json_object",
                "schema": llm_match_list_schema(),
            },
            max_tokens=llm_cfg.max_tokens * max_matches,
            temperature=llm_cfg.temperature,
        ),
    )
    raw = cast(str | None, response["choices"][0]["message"]["content"])
    if not raw:
        raise LLMStructuredOutputError("LLM returned empty content")
    return _parse_llm_match_list(raw).matches


def _to_result_match(
    chunk: Any,
    *,
    confidence: Any = "UNKNOWN",
    rationale: str = "Not provided.",
    co_techniques: list[CoTechnique] | None = None,
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
        confidence=confidence,
        rationale=rationale,
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


def _rerank_matches(
    matches: list[Any],
    top_n: int,
) -> list[TechniqueMatch]:
    from .config import get_config

    cfg = get_config()
    system = cfg.prompts.system
    subtech_map = _get_subtech_map()
    reranked: list[TechniqueMatch] = []

    chunks_to_process = matches[:top_n]
    for i, chunk in enumerate(chunks_to_process):
        candidates = [
            {
                "mitre_id": chunk.technique_id,
                "name": chunk.name,
                "tactic": chunk.tactic,
                "description": chunk.description,
                "score": round(chunk.score, 4),
            }
        ]
        for co in chunk.co_techniques:
            candidates.append(
                {
                    "mitre_id": co.technique_id,
                    "name": co.name,
                    "tactic": co.tactic,
                    "description": co.description,
                    "score": round(co.score, 4),
                }
            )

        remaining_chunks = len(chunks_to_process) - 1 - i
        slots_needed = max(1, (top_n - len(reranked)) - remaining_chunks)

        try:
            llm_matches = _llm_call_multi(
                [
                    {"role": "system", "content": system},
                    {
                        "role": "user",
                        "content": build_prompt(
                            chunk.chunk_text, candidates, slots_needed
                        ),
                    },
                ],
                slots_needed,
            )
        except LLMStructuredOutputError as exc:
            _log.warning(
                "LLM reranking failed for chunk %s: %s", chunk.technique_id, exc
            )
            reranked.append(_to_result_match(chunk))
            reranked[-1].color_idx = len(reranked) - 1
            continue

        if not llm_matches:
            reranked.append(_to_result_match(chunk))
            reranked[-1].color_idx = len(reranked) - 1
            continue

        selected_ids = {tm.technique_id for tm in llm_matches}
        unselected = []
        for c in candidates:
            if c["mitre_id"] not in selected_ids:
                unselected.append(
                    {
                        "technique_id": str(c["mitre_id"]),
                        "name": str(c["name"]),
                        "description": str(c["description"]),
                        "tactic": str(c["tactic"]),
                        "score": float(c["score"]),
                    }
                )

        for j, tm in enumerate(llm_matches):
            orig_id = tm.technique_id
            subtechs = subtech_map.get(tm.technique_id, [])
            if subtechs:
                try:
                    tm = _llm_call(
                        [
                            {"role": "system", "content": system},
                            {
                                "role": "user",
                                "content": build_subtechnique_prompt(
                                    chunk.chunk_text, tm, subtechs
                                ),
                            },
                        ]
                    )
                except LLMStructuredOutputError as exc:
                    _log.warning(
                        "LLM subtechnique refinement failed for %s: %s",
                        tm.technique_id,
                        exc,
                    )

            orig = next((c for c in candidates if c["mitre_id"] == orig_id), None)
            updated_chunk = dataclasses.replace(
                chunk,
                technique_id=tm.technique_id,
                name=tm.name,
                description=str(orig["description"]) if orig else chunk.description,
                tactic=str(orig["tactic"]) if orig else chunk.tactic,
                score=float(orig["score"]) if orig else chunk.score,
                color_idx=len(reranked),
            )
            reranked.append(
                _to_result_match(
                    updated_chunk,
                    confidence=tm.confidence,
                    rationale=tm.rationale,
                    co_techniques=(
                        [
                            CoTechnique(
                                technique_id=co["technique_id"],
                                name=co["name"],
                                tactic=co["tactic"],
                                score=co["score"],
                            )
                            for co in unselected
                        ]
                        if j == 0
                        else []
                    ),
                )
            )

    return reranked


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
    from .config import get_config

    top_k = validate_top_k(top_k)

    top_scan = top_k * 2
    _log.info("Scanning text against MITRE ATT&CK techniques (top %d)...", top_scan)
    scan_result: ScanResult = scan_text(query, top_scan)
    if not scan_result.matches:
        return SearchResult(text=query, matches=[])

    _log.info("Running LLM to rerank chunk-level candidates...")
    resolved = _rerank_matches(scan_result.matches, top_k)

    if not all_confidence:
        _log.info("Filtering matches to only HIGH or CERTAIN confidence...")
        resolved = [m for m in resolved if m.confidence in ("HIGH", "CERTAIN")]
        for idx, match in enumerate(resolved):
            resolved[idx].color_idx = idx

    return SearchResult(text=query, matches=resolved[:top_k])
