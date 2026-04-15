"""
Harrier embedding model wrapper.

Two public functions:
    embed_query(text)          → list[float]  (1024-dim, L2-normalised, query prefix applied)
    embed_documents(texts)     → np.ndarray   (N × 1024, L2-normalised, no prefix)

The model is downloaded automatically from HuggingFace Hub on first call
and cached in ~/.cache/huggingface/hub/.
"""

import logging

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_ID = "microsoft/harrier-oss-v1-0.6b"
# Harrier is instruction-aware for queries; documents are encoded without a prefix.
_QUERY_PREFIX = "Retrieve semantically similar text: "

_model: SentenceTransformer | None = None
_log = logging.getLogger(__name__)


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _log.info("Loading embedding model %s", MODEL_ID)
        # Suppress the safetensors/transformers tqdm progress bar during weight loading
        import transformers
        transformers.logging.set_verbosity_error()
        transformers.logging.disable_progress_bar()
        _model = SentenceTransformer(MODEL_ID)
    return _model


def embed_query(text: str) -> list[float]:
    """Embed a single query string. Returns a 1024-dim L2-normalised vector."""
    vec = _get_model().encode(
        [_QUERY_PREFIX + text],
        normalize_embeddings=True,
    )
    return vec[0].tolist()


def embed_documents(
    texts: list[str],
    batch_size: int = 32,
    show_progress: bool = False,
) -> np.ndarray:
    """Embed a list of document strings. Returns float32 array of shape (N, 1024)."""
    return _get_model().encode(
        texts,
        normalize_embeddings=True,
        batch_size=batch_size,
        show_progress_bar=show_progress,
    )
