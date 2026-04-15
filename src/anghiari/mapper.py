"""
Core mapping logic: embed → vector search → LLM phase 1 → LLM phase 2 (subtechnique).

Both models are downloaded automatically from HuggingFace Hub on first use
and cached in ~/.cache/huggingface/hub/.
"""

import json
import sys
from pathlib import Path

import chromadb
from llama_cpp import Llama

from .embedder import embed_query
from .models import SearchResult, TechniqueMatch
from .prompt import SYSTEM, build_prompt, build_subtechnique_prompt

# ── Paths (relative to working directory) ────────────────────────────────────
DATA_DIR = Path("data")
CHROMA_DIR = DATA_DIR / "chroma_db"
SUBTECH_MAP_FILE = DATA_DIR / "subtechnique_map.json"
COLLECTION_NAME = "mitre_techniques"

# ── LLM — downloaded automatically from HuggingFace Hub ─────────────────────
LLM_REPO_ID = "nvidia/NVIDIA-Nemotron-3-Nano-4B-GGUF"
LLM_FILENAME = "NVIDIA-Nemotron3-Nano-4B-Q4_K_M.gguf"

# ── Lazy singletons ───────────────────────────────────────────────────────────
_collection = None
_llm: Llama | None = None
_subtech_map: dict[str, list[dict]] | None = None


def _get_collection():
    global _collection
    if _collection is None:
        if not CHROMA_DIR.exists():
            raise RuntimeError(
                f"ChromaDB index not found at {CHROMA_DIR}. Run `anghiari index` first."
            )
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _collection = client.get_collection(COLLECTION_NAME)
    return _collection


def _get_llm() -> Llama:
    global _llm
    if _llm is None:
        print(f"Loading LLM ({LLM_REPO_ID} / {LLM_FILENAME})...", file=sys.stderr)
        _llm = Llama.from_pretrained(
            repo_id=LLM_REPO_ID,
            filename=LLM_FILENAME,
            n_ctx=131072,
            n_gpu_layers=-1,  # Metal: offload all layers to GPU on Apple Silicon
            verbose=False,
            chat_format="chatml",
        )
    return _llm


def _get_subtech_map() -> dict[str, list[dict]]:
    global _subtech_map
    if _subtech_map is None:
        if not SUBTECH_MAP_FILE.exists():
            raise RuntimeError(
                f"Subtechnique map not found at {SUBTECH_MAP_FILE}. "
                "Run `anghiari index` first."
            )
        _subtech_map = json.loads(SUBTECH_MAP_FILE.read_text())
    return _subtech_map


def _llm_call(messages: list[dict]) -> TechniqueMatch:
    response = _get_llm().create_chat_completion(
        messages=messages,
        response_format={"type": "json_object"},
        max_tokens=512,
        temperature=0.1,
    )
    raw = response["choices"][0]["message"]["content"]
    return TechniqueMatch.model_validate_json(raw)


def search_technique(query: str, top_k: int = 5) -> SearchResult:
    """Map a free-text attack description to the best-matching MITRE ATT&CK technique."""
    # ── Phase 0: embed the query ─────────────────────────────────────────────
    query_vec = embed_query(query)

    # ── Phase 1: vector search ───────────────────────────────────────────────
    results = _get_collection().query(query_embeddings=[query_vec], n_results=top_k)
    candidates = [
        {**meta, "score": round(1.0 - dist, 4)}
        for meta, dist in zip(results["metadatas"][0], results["distances"][0])
    ]

    # ── Phase 2: LLM picks best from top-k ──────────────────────────────────
    best = _llm_call(
        [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": build_prompt(query, candidates)},
        ]
    )

    # ── Phase 3: subtechnique resolution ────────────────────────────────────
    subtechs = _get_subtech_map().get(best.technique_id, [])
    if subtechs:
        refined = _llm_call(
            [
                {"role": "system", "content": SYSTEM},
                {
                    "role": "user",
                    "content": build_subtechnique_prompt(query, best, subtechs),
                },
            ]
        )
        best = refined

    return SearchResult(best_match=best, candidates=candidates)
