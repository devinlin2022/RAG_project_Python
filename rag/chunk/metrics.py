"""Zero-cost structural metrics.

These are not retrieval precision. They measure whether a chunker respected the
document's own structure, which can be computed in milliseconds without a gold
set and which is exactly what matters for contracts:

- boundary: does a chunk end where a sentence ends?
- clause:   is every 第X条 contained in one chunk?
- size:     are chunk lengths inside the target window?
- section:  does a chunk know which heading it belongs to?

Real hit@5 / MRR need labelled questions and arrive in V6.
"""

import re
from collections import Counter

from rag.chunk.types import Chunk

_CLAUSE = re.compile(r"第\s*[一二三四五六七八九十百千零〇两\d]+\s*条")
_TERMINAL = "。！？；：!?;:.）)】」』”"
_GRAM = 8


def boundary_violation(chunks: list[Chunk]) -> float:
    """Share of prose chunks not ending on a terminal punctuation mark.

    Tables and figure captions are excluded: a Markdown row ends with "|" by
    construction, so counting them would report a defect that does not exist.
    """
    prose = [c for c in chunks if c.kind == "text" and c.text.strip()]
    if not prose:
        return 0.0
    bad = sum(1 for c in prose if c.text.strip()[-1] not in _TERMINAL)
    return round(bad / len(prose), 4)


def clause_integrity(source_text: str, chunks: list[Chunk]) -> float:
    """Share of 第X条 labels that survive whole inside a single chunk."""
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

    Tables are excluded: their header and separator rows repeat by design, and
    that repetition is what makes a split table readable.
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


def composite(chunks: list[Chunk], source_text: str, min_chars: int, max_chars: int) -> dict:
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
    score = (
        metrics["clause_integrity"] * 0.45
        + (1 - metrics["boundary_violation"]) * 0.25
        + metrics["size_conformance"] * 0.20
        + metrics["section_coverage"] * 0.10
    )
    metrics["struct_score"] = round(score - metrics["overlap_waste"] * 0.05, 4)
    return metrics
