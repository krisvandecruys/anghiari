"""
Core mapping logic: embed → vector search → LLM phase 1 → LLM phase 2 (subtechnique).

Both models are downloaded automatically from HuggingFace Hub on first use
and cached in ~/.cache/huggingface/hub/.
"""

import contextlib
import json
import logging
import os
import warnings

import chromadb
from llama_cpp import Llama

from .embedder import embed_query
from .models import LLMMatchList, SearchResult, TechniqueMatch
from .prompt import build_prompt, build_subtechnique_prompt

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


def _llm_call(messages: list[dict]) -> TechniqueMatch:
    from .config import get_config

    llm_cfg = get_config().llm
    response = _get_llm().create_chat_completion(
        messages=messages,
        response_format={"type": "json_object"},
        max_tokens=llm_cfg.max_tokens,
        temperature=llm_cfg.temperature,
    )
    raw = response["choices"][0]["message"]["content"]
    return TechniqueMatch.model_validate_json(raw)


def _llm_call_multi(messages: list[dict], max_matches: int) -> list[TechniqueMatch]:
    from .config import get_config

    llm_cfg = get_config().llm
    response = _get_llm().create_chat_completion(
        messages=messages,
        response_format={"type": "json_object"},
        max_tokens=llm_cfg.max_tokens * max_matches,
        temperature=llm_cfg.temperature,
    )
    raw = response["choices"][0]["message"]["content"]
    return LLMMatchList.model_validate_json(raw).matches


def search_technique(
    query: str, top_k: int | None = None, all_confidence: bool = False
) -> SearchResult:
    """Map a free-text attack description to one or more MITRE ATT&CK techniques."""
    from .config import get_config

    cfg = get_config()
    if top_k is None:
        top_k = cfg.search.top_k
    max_matches = min(cfg.search.max_matches, top_k)
    system = cfg.prompts.system

    # ── Phase 0: embed the query ─────────────────────────────────────────────
    _log.info("Embedding full query text for generic RAG search...")
    query_vec = embed_query(query)

    # ── Phase 1: vector search ───────────────────────────────────────────────
    _log.info("Querying ChromaDB vector index for top %d candidates...", top_k)
    results = _get_collection().query(query_embeddings=[query_vec], n_results=top_k)
    candidates = [
        {**meta, "score": round(1.0 - dist, 4)}
        for meta, dist in zip(results["metadatas"][0], results["distances"][0])
    ]

    # ── Phase 2: LLM picks 1–max_matches from top-k ─────────────────────────
    _log.info("Running LLM to rerank top candidates...")
    matches = _llm_call_multi(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": build_prompt(query, candidates, max_matches)},
        ],
        max_matches,
    )

    # ── Phase 3: subtechnique resolution for each match ─────────────────────
    subtech_map = _get_subtech_map()
    resolved = []
    _log.info("Resolving exact subtechniques where necessary...")
    for match in matches:
        subtechs = subtech_map.get(match.technique_id, [])
        if subtechs:
            refined = _llm_call(
                [
                    {"role": "system", "content": system},
                    {
                        "role": "user",
                        "content": build_subtechnique_prompt(query, match, subtechs),
                    },
                ]
            )
            resolved.append(refined)
        else:
            resolved.append(match)

    if not all_confidence:
        _log.info("Filtering matches to only HIGH or CERTAIN confidence...")
        resolved = [m for m in resolved if m.confidence in ("HIGH", "CERTAIN")]

    return SearchResult(matches=resolved, candidates=candidates)
