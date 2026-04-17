"""
Chunk-level ATT&CK technique scanner with source-text highlighting.

Splits input text into overlapping sentence windows, embeds each chunk with
Harrier's query encoder, then scores all chunks against every stored technique
via a single matrix multiply.  No LLM is needed.

Public API:
    scan_text(text, top_n=8)  -> ScanResult
    render(result)            -> str  (ANSI-coloured terminal output)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

# ── ANSI colour table (8 terminal-safe colours) ───────────────────────────────

_COLORS = [
    "\033[91m",  # bright red
    "\033[92m",  # bright green
    "\033[93m",  # bright yellow
    "\033[94m",  # bright blue
    "\033[95m",  # bright magenta
    "\033[96m",  # bright cyan
    "\033[33m",  # dark yellow / orange
    "\033[97m",  # bright white
]
_RESET = "\033[0m"
_BOLD = "\033[1m"
_LINE = "─" * 62


# ── Data shapes ───────────────────────────────────────────────────────────────


@dataclass
class CoTechnique:
    """A technique that co-fires on the same chunk as a primary ChunkMatch."""

    technique_id: str
    name: str
    tactic: str
    score: float


@dataclass
class ChunkMatch:
    technique_id: str
    name: str
    tactic: str
    score: float  # cosine similarity 0–1
    chunk_text: str  # sentence window that best matched this technique
    start: int  # char offset in original text (inclusive)
    end: int  # char offset in original text (exclusive)
    color_idx: int  # index into _COLORS
    co_techniques: list[CoTechnique] = field(default_factory=list)
    confidence: str | None = None  # set by CLI after LLM reranking
    rationale: str | None = None  # set by CLI after LLM reranking


@dataclass
class ScanResult:
    matches: list[ChunkMatch]  # top N, score-descending
    text: str  # original input (used by render())


# ── Lazy singleton: technique embedding matrix ────────────────────────────────

_tech_matrix: np.ndarray | None = None
_tech_meta: list[dict] | None = None


def _get_tech_matrix() -> tuple[np.ndarray, list[dict]]:
    """Load all technique embeddings from ChromaDB once and cache in memory."""
    global _tech_matrix, _tech_meta
    if _tech_matrix is None:
        from .mapper import _get_collection

        result = _get_collection().get(include=["embeddings", "metadatas"])
        _tech_meta = result["metadatas"]
        _tech_matrix = np.array(result["embeddings"], dtype=np.float32)
    assert _tech_matrix is not None and _tech_meta is not None
    return _tech_matrix, _tech_meta


# ── Span and technique helpers ────────────────────────────────────────────────

_CO_FIRE_THRESHOLD = 0.05  # max score gap vs primary to qualify as a co-fire
_MAX_CO_FIRES = 4  # max number of co-techniques to attach to a primary match


def _span_overlaps(
    a_start: int, a_end: int, b_start: int, b_end: int, threshold: float = 0.4
) -> bool:
    """Return True if two char spans overlap by at least *threshold* of the shorter span."""
    overlap = max(0, min(a_end, b_end) - max(a_start, b_start))
    shorter = min(a_end - a_start, b_end - b_start)
    return (overlap / shorter) >= threshold if shorter > 0 else False


def _is_parent_sub_pair(id_a: str, id_b: str) -> bool:
    """True if one ID is the parent of the other (same base, different specificity)."""
    return id_a.split(".")[0] == id_b.split(".")[0] and id_a != id_b


def _is_subtechnique(technique_id: str) -> bool:
    return "." in technique_id


# ── Chunking ──────────────────────────────────────────────────────────────────

_MIN_CHUNK_LEN = 20  # ignore fragments shorter than this


def multi_level_chunks(text: str) -> list[tuple[str, int, int]]:
    """Return list of (chunk_text, start, end) for the input text.

    Generates a rich set of overlapping spans:
    1. Paragraphs (split by blank lines)
    2. Sentences (split by sentence-ending punctuation)
    3. Overlapping 2-sentence windows
    4. Subsentences (split by commas, colons, semicolons)
    5. Quoted text (text between quotes)

    Character offsets are exact slices of *text*:  text[start:end] == chunk_text.
    This allows the greedy selector to pick the tightest span that strongly
    matches a technique, falling back to sentences or paragraphs if needed.
    """
    chunks: set[tuple[str, int, int]] = set()

    def add_chunk(start: int, end: int):
        window = text[start:end].strip()
        if len(window) >= _MIN_CHUNK_LEN:
            # Re-adjust start and end to the stripped window
            l_strip_len = len(text[start:end]) - len(text[start:end].lstrip())
            r_strip_len = len(text[start:end]) - len(text[start:end].rstrip())
            s = start + l_strip_len
            e = end - r_strip_len
            if e > s and len(text[s:e]) >= _MIN_CHUNK_LEN:
                chunks.add((text[s:e], s, e))

    def split_span(start: int, end: int, pattern: re.Pattern) -> list[tuple[int, int]]:
        span_text = text[start:end]
        parts = []
        pos = 0
        for match in pattern.finditer(span_text):
            parts.append((start + pos, start + match.start()))
            pos = match.end()
        parts.append((start + pos, end))

        result = []
        for s, e in parts:
            if e > s:
                result.append((s, e))
                add_chunk(s, e)
        return result

    # 1. Paragraphs
    para_spans = []
    pos = 0
    _PARA_SPLIT = re.compile(r"\n+")
    for match in _PARA_SPLIT.finditer(text):
        if match.start() > pos:
            para_spans.append((pos, match.start()))
            add_chunk(pos, match.start())
        pos = match.end()
    if pos < len(text):
        para_spans.append((pos, len(text)))
        add_chunk(pos, len(text))

    _SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
    _SUB_SPLIT = re.compile(r"(?<=[,;:])\s+")
    _QUOTE_EXTRACT = re.compile(r'"([^"]+)"|\'([^\']+)\'')

    # 2. Sentences
    for p_start, p_end in para_spans:
        sent_spans = split_span(p_start, p_end, _SENT_SPLIT)

        # 3. Overlapping 2-sentence windows
        for i in range(len(sent_spans) - 1):
            s1_start, _ = sent_spans[i]
            _, s2_end = sent_spans[i + 1]
            add_chunk(s1_start, s2_end)

        # 4. Subsentences
        for s_start, s_end in sent_spans:
            split_span(s_start, s_end, _SUB_SPLIT)

            # 5. Quoted text inside sentence
            sent_text = text[s_start:s_end]
            for match in _QUOTE_EXTRACT.finditer(sent_text):
                add_chunk(s_start + match.start(), s_start + match.end())

    # Sort chunks by start, then -end to be deterministic
    return sorted(list(chunks), key=lambda x: (x[1], -x[2]))


# ── Core scoring ──────────────────────────────────────────────────────────────


def scan_text(text: str, top_n: int = 8) -> ScanResult:
    """Score every sentence chunk against every stored ATT&CK technique.

    Returns up to *top_n* ChunkMatches, each from a distinct (non-overlapping)
    region of the source text.  When two techniques score nearly equally on the
    same chunk, the lower-scoring one is recorded as a co-technique on the
    primary match rather than consuming a separate slot.
    """
    from .config import get_config
    from .embedder import embed_documents

    chunks = multi_level_chunks(text)
    if not chunks:
        return ScanResult(matches=[], text=text)

    prefix = get_config().embedder.query_prefix
    chunk_texts = [c[0] for c in chunks]

    # Embed chunks with the query prefix — they are queries against the corpus.
    chunk_matrix = embed_documents(
        [prefix + t for t in chunk_texts]
    )  # (n_chunks, 1024)

    tech_matrix, tech_meta = _get_tech_matrix()  # (n_techs, 1024)

    # Cosine similarity: vectors are L2-normalised, so dot product == cosine.
    scores = chunk_matrix @ tech_matrix.T  # (n_chunks, n_techs)

    best_chunk_idxs = scores.argmax(axis=0)  # (n_techs,)
    best_scores = scores.max(axis=0)  # (n_techs,)

    # ── Greedy selection with co-firing detection ─────────────────────────────
    # Walk techniques best-score-first.  For each technique:
    #   • No overlap with any claimed span → new primary match, claim the span.
    #   • Overlaps a claimed span, same parent/sub family → subtechnique wins
    #     (upgrade the primary if the new one is more specific).
    #   • Overlaps a claimed span, genuinely distinct, score gap ≤ threshold →
    #     record as a co-technique on the primary (same colour, co-mentioned).
    sorted_tech_idxs = best_scores.argsort()[::-1]
    claimed: list[ChunkMatch] = []
    claimed_spans: list[tuple[int, int]] = []
    seen_ids: set[str] = (
        set()
    )  # all technique IDs already placed (primary or co-technique)

    for tech_idx in sorted_tech_idxs:
        if len(claimed) >= top_n and all(
            best_scores[tech_idx] < claimed[i].score - _CO_FIRE_THRESHOLD
            for i in range(len(claimed))
        ):
            break  # nothing left can qualify as primary or co-fire

        ci = int(best_chunk_idxs[tech_idx])
        cs, ce = chunks[ci][1], chunks[ci][2]
        tech_score = float(best_scores[tech_idx])
        meta = tech_meta[tech_idx]
        tech_id = meta["mitre_id"]

        # Never place the same technique ID twice anywhere in the output.
        if tech_id in seen_ids:
            continue

        overlap_idx = next(
            (
                i
                for i, (s, e) in enumerate(claimed_spans)
                if _span_overlaps(cs, ce, s, e)
            ),
            None,
        )

        if overlap_idx is None:
            if len(claimed) < top_n:
                claimed.append(
                    ChunkMatch(
                        technique_id=tech_id,
                        name=meta["name"],
                        tactic=meta.get("tactic", ""),
                        score=tech_score,
                        chunk_text=chunks[ci][0],
                        start=cs,
                        end=ce,
                        color_idx=len(claimed),
                    )
                )
                claimed_spans.append((cs, ce))
                seen_ids.add(tech_id)
        else:
            primary = claimed[overlap_idx]
            score_gap = primary.score - tech_score

            if _is_parent_sub_pair(tech_id, primary.technique_id):
                if _is_subtechnique(tech_id) and not _is_subtechnique(
                    primary.technique_id
                ):
                    # Upgrade: replace parent with the more specific subtechnique.
                    seen_ids.discard(primary.technique_id)
                    primary.technique_id = tech_id
                    primary.name = meta["name"]
                    primary.tactic = meta.get("tactic", "")
                    primary.score = tech_score
                    seen_ids.add(tech_id)
                # else: primary is already subtechnique or same specificity → skip.
            elif (
                score_gap <= _CO_FIRE_THRESHOLD
                and len(primary.co_techniques) < _MAX_CO_FIRES
            ):
                # Genuinely distinct technique, nearly as strong → co-mention.
                primary.co_techniques.append(
                    CoTechnique(
                        technique_id=tech_id,
                        name=meta["name"],
                        tactic=meta.get("tactic", ""),
                        score=tech_score,
                    )
                )
                seen_ids.add(tech_id)

    return ScanResult(matches=claimed, text=text)


# ── Rendering ─────────────────────────────────────────────────────────────────


def _color(idx: int) -> str:
    return _COLORS[idx % len(_COLORS)]


def render(result: ScanResult) -> str:
    """Produce a two-block ANSI terminal string:
    1. Ranked table of matches with triggering excerpts and co-techniques.
    2. Original text with matched spans highlighted in colour.
    """
    lines: list[str] = []

    # ── Block 1: ranked table ─────────────────────────────────────────────────
    lines.append(f"\n{_BOLD}TECHNIQUE SCAN{_RESET}  ({len(result.matches)} matches)")
    lines.append(_LINE)

    for m in result.matches:
        c = _color(m.color_idx)
        label = f"[{m.color_idx + 1}]"
        excerpt = m.chunk_text[:120].replace("\n", " ")
        if len(m.chunk_text) > 120:
            excerpt += "…"
        conf_badge = f"  [{m.confidence}]" if m.confidence else ""
        lines.append(
            f"{c}{_BOLD}{label}{_RESET} {m.score:.3f}  "
            f"{m.technique_id:<12} {m.name:<40} {m.tactic}{conf_badge}"
        )
        for co in m.co_techniques:
            lines.append(
                f"         {c}also: {co.technique_id:<12} {co.name}  ({co.score:.3f}){_RESET}"
            )
        lines.append(f'    {c}↳ "{excerpt}"{_RESET}')
        if m.rationale:
            lines.append(f"    {c}↳ {m.rationale}{_RESET}")
        lines.append("")

    # ── Block 2: annotated source text ────────────────────────────────────────
    lines.append(f"{_BOLD}ANNOTATED TEXT{_RESET}")
    lines.append(_LINE)

    # Build a per-character list of all colour indices that cover each position.
    # Overlapping spans accumulate multiple entries; the render cycles through them.
    text = result.text
    char_colors: list[list[int]] = [[] for _ in range(len(text))]
    for m in result.matches:
        for i in range(m.start, min(m.end, len(text))):
            char_colors[i].append(m.color_idx)

    def _effective_color(i: int) -> int | None:
        colors = char_colors[i]
        if not colors:
            return None
        # Single technique: use its colour directly.
        # Multiple techniques overlapping: cycle through their colours by position,
        # producing a letter-by-letter alternation (e.g. RED YEL RED YEL …).
        return colors[i % len(colors)]

    # Emit text, inserting ANSI codes whenever the effective colour changes.
    buf: list[str] = []
    current: int | None = -1  # sentinel: not yet started
    for i, ch in enumerate(text):
        ec = _effective_color(i)
        if ec != current:
            if current is not None and current >= 0:
                buf.append(_RESET)
            if ec is not None:
                buf.append(_color(ec))
            current = ec
        buf.append(ch)
    if current is not None and current >= 0:
        buf.append(_RESET)

    lines.append("".join(buf))
    lines.append(_LINE)

    return "\n".join(lines)


# ── Standalone entry point ────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import sys

    p = argparse.ArgumentParser(
        description="Scan a threat description against MITRE ATT&CK techniques."
    )
    p.add_argument("text", nargs="?", help="Text to scan (omit to read from stdin)")
    p.add_argument("--file", "-f", metavar="FILE", help="Read text from a file")
    p.add_argument(
        "--top", "-n", type=int, default=8, help="Number of top techniques (default: 8)"
    )
    args = p.parse_args()

    if args.file:
        blob = open(args.file).read()
    elif args.text:
        blob = args.text
    else:
        blob = sys.stdin.read()

    print(render(scan_text(blob, args.top)))
