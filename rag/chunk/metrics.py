"""Structural proxy metrics for chunking quality."""

import re
from collections import Counter

from rag.chunk.types import Chunk

_CLAUSE = re.compile(r"第\s*[一二三四五六七八九十百千零〇两\d]+\s*条")
_TERMINAL = "。！？；：!?;:.）)】」』”"
_GRAM = 8


def boundary_violation(chunks: list[Chunk]) -> float:
    """Share of prose chunks not ending on terminal punctuation.

    Tables and captions are excluded: a Markdown row ends with a pipe by design.
    """
    prose = [c for c in chunks if c.kind == "text" and c.text.strip()]
    if not prose:
        return 0.0
    bad = sum(1 for c in prose if c.text.strip()[-1] not in _TERMINAL)
    return round(bad / len(prose), 4)


def clause_integrity(source_text: str, chunks: list[Chunk]) -> float:
    expected = Counter(_CLAUSE.findall(source_text))
    if not expected:
        return 1.0
    found: Counter = Counter()
    for chunk in chunks:
        found.update(_CLAUSE.findall(chunk.text))
    intact = sum(min(count, found.get(label, 0)) for label, count in expected.items())
    return round(intact / sum(expected.values()), 4)


def size_conformance(chunks: list[Chunk], min_chars: int, max_chars: int) -> float:
    if not chunks:
        return 0.0
    ok = sum(1 for c in chunks if min_chars * 0.5 <= c.chars <= max_chars * 1.2)
    return round(ok / len(chunks), 4)


def section_coverage(chunks: list[Chunk]) -> float:
    if not chunks:
        return 0.0
    ok = sum(1 for c in chunks if c.section and c.section != "（正文）")
    return round(ok / len(chunks), 4)


def overlap_waste(chunks: list[Chunk]) -> float:
    """Share of 8-grams repeated across prose chunks. High means redundancy.

    Tables are excluded: header and separator rows repeat by design.
    """
    prose = [c for c in chunks if c.kind == "text"]
    if len(prose) < 2:
        return 0.0
    total = 0
    seen: Counter = Counter()
    for chunk in prose:
        text = re.sub(r"\s+", "", chunk.text)
        grams = [text[i : i + _GRAM] for i in range(0, max(len(text) - _GRAM + 1, 1))]
        total += len(grams)
        seen.update(set(grams))
    if total == 0:
        return 0.0
    unique = len(seen)
    return round(1 - unique / total, 4)


def prose_ratio(text: str) -> float:
    """Share of lines ending on sentence punctuation.

    A bullet deck scores near zero, so the metric is reported as not applicable
    and dropped from the score instead of penalising the document.
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return 0.0
    ended = sum(1 for line in lines if line[-1] in "。！？；.!?;")
    return round(ended / len(lines), 4)


def typed_preservation(chunks: list[Chunk], expected_typed: int) -> float | None:
    """Share of table, figure and OCR blocks kept as their own chunk type.

    Only structure-aware strategies can keep a table a table.
    """
    if expected_typed <= 0:
        return None
    kept = sum(1 for chunk in chunks if chunk.kind != "text")
    return round(min(kept / expected_typed, 1.0), 4)


def composite(
    chunks: list[Chunk],
    source_text: str,
    min_chars: int,
    max_chars: int,
    expected_typed: int = 0,
) -> dict:
    """Bundle the proxies into one comparable score, weighted for clause work."""
    metrics = {
        "chunks": len(chunks),
        "avg_chars": round(sum(c.chars for c in chunks) / len(chunks)) if chunks else 0,
        "chunks_per_1k": round(len(chunks) / max(len(source_text), 1) * 1000, 2),
        "boundary_violation": boundary_violation(chunks),
        "clause_integrity": clause_integrity(source_text, chunks),
        "size_conformance": size_conformance(chunks, min_chars, max_chars),
        "section_coverage": section_coverage(chunks),
        "overlap_waste": overlap_waste(chunks),
    }
    metrics["prose_ratio"] = prose_ratio(source_text)
    metrics["boundary_applicable"] = metrics["prose_ratio"] >= 0.2
    metrics["clause_applicable"] = bool(_CLAUSE.search(source_text))
    metrics["typed_preservation"] = typed_preservation(chunks, expected_typed)
    metrics["typed_expected"] = expected_typed

    # Only applicable terms enter the score, and their weights are renormalised.
    # A document without clauses, or without sentences, must not be graded on them.
    terms: list[tuple[float, float]] = []
    if metrics["clause_applicable"]:
        terms.append((metrics["clause_integrity"], 0.45))
    if metrics["boundary_applicable"]:
        terms.append((1 - metrics["boundary_violation"], 0.25))
    if metrics["typed_preservation"] is not None:
        terms.append((metrics["typed_preservation"], 0.30))
    terms.append((metrics["size_conformance"], 0.20))
    terms.append((metrics["section_coverage"], 0.10))

    weight = sum(w for _, w in terms)
    score = sum(value * w for value, w in terms) / weight if weight else 0.0
    metrics["struct_score"] = round(score - metrics["overlap_waste"] * 0.05, 4)
    return metrics
