"""Semantic chunking: break where meaning changes.

Breakpoints come from a percentile of neighbour distances, not a fixed threshold:
the distance scale depends on the embedding model.
"""

import logging
import re

from rag.chunk.types import Chunk
from rag.embed.client import EmbedError, embed_texts

log = logging.getLogger(__name__)

SENTENCE_END = re.compile(r"(?<=[。！？!?])|(?<=[；;])(?=.{8,})|\n+")
# "1." and "No." must not become sentence boundaries.
_DECIMAL = re.compile(r"(?<=\d)\.(?=\d)")
MAX_SENTENCES = 1200


def split_sentences(text: str) -> list[str]:
    """Sentence split that tolerates Chinese punctuation and decimals."""
    guarded = _DECIMAL.sub("·", text)
    pieces = [piece.strip() for piece in SENTENCE_END.split(guarded)]
    return [piece.replace("·", ".") for piece in pieces if piece]


def _group(spans: list[str], size: int, min_chars: int) -> list[str]:
    """Merge sentences into chunks honouring the size window."""
    out: list[str] = []
    current = ""
    for span in spans:
        if not current:
            current = span
        elif len(current) + len(span) <= size:
            current += span
        else:
            out.append(current)
            current = span
    if current:
        out.append(current)

    merged: list[str] = []
    for piece in out:
        if merged and len(piece) < min_chars and len(merged[-1]) + len(piece) <= size:
            merged[-1] += piece
        else:
            merged.append(piece)
    return merged


def semantic_split(
    text: str,
    size: int,
    min_chars: int,
    percentile: float,
) -> tuple[list[str], dict]:
    """Split on semantic breakpoints. Falls back to sentence packing on error."""
    sentences = split_sentences(text)
    if len(sentences) < 3:
        return _group(sentences, size, min_chars), {"mode": "short", "sentences": len(sentences)}
    if len(sentences) > MAX_SENTENCES:
        return _group(sentences, size, min_chars), {
            "mode": "too_many_sentences",
            "sentences": len(sentences),
        }

    try:
        vectors = embed_texts(sentences)
    except EmbedError as exc:
        log.warning("semantic split degraded to sentence packing: %s", exc)
        return _group(sentences, size, min_chars), {"mode": "fallback", "error": str(exc)}

    import numpy as np

    matrix = np.asarray(vectors, dtype="float32")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = matrix / np.clip(norms, 1e-8, None)
    sims = np.sum(matrix[:-1] * matrix[1:], axis=1)
    distances = 1.0 - sims

    threshold = float(np.percentile(distances, percentile))
    breakpoints = {i + 1 for i, value in enumerate(distances) if value >= threshold}

    spans: list[str] = []
    current = sentences[0]
    for index in range(1, len(sentences)):
        if index in breakpoints:
            spans.append(current)
            current = sentences[index]
        else:
            current += sentences[index]
    spans.append(current)

    return _group(spans, size, min_chars), {
        "mode": "percentile",
        "sentences": len(sentences),
        "breakpoints": len(breakpoints),
        "threshold": round(threshold, 4),
        "percentile": percentile,
    }


def chunk_semantic(
    blocks: list[tuple[int, str]],
    size: int,
    overlap: int,
    min_chars: int,
    percentile: float,
) -> tuple[list[Chunk], dict]:
    """Semantic chunks over the whole document, structure deliberately ignored."""
    if not blocks:
        return [], {"mode": "empty"}

    pages = [page for page, _ in blocks]
    text = "\n".join(block for _, block in blocks)
    pieces, info = semantic_split(text, size, min_chars, percentile)

    chunks: list[Chunk] = []
    cursor = 0
    for index, piece in enumerate(pieces):
        offset = text.find(piece.strip()[:60], cursor)
        if offset < 0:
            offset = cursor
        else:
            cursor = offset
        ratio = offset / max(len(text), 1)
        page = pages[min(int(ratio * len(pages)), len(pages) - 1)]
        chunks.append(
            Chunk(
                idx=index,
                text=piece.strip(),
                page_from=page,
                page_to=page,
                kind="text",
                strategy="semantic",
            )
        )
    return chunks, info
