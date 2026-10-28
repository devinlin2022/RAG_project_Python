"""Evaluation metrics: retrieval hit rate, MRR, citation grounding, abstention.

All functions are pure so they can be tested without a model or a PDF.
"""

import re

_CITE = re.compile(r"\[(\d+)\]")
_ABSTAIN = ("未提及", "无法回答", "没有相关", "未找到", "不包含", "无法确定", "资料中没有")


def rank_of_first_hit(
    spans: list[tuple[int, int]], gold: list[int], k: int | None = None
) -> int:
    """1-based rank of the first hit, 0 when nothing matches.

    A chunk spans a page range, so a gold page inside that range counts: a
    four-page chunk labelled p.7-10 does contain what is printed on p.9.
    """
    if not gold:
        return 0
    wanted = set(gold)
    for rank, (first, last) in enumerate(spans[:k] if k else spans, start=1):
        if wanted & set(range(first, last + 1)):
            return rank
    return 0


def hit_at_k(pages: list[int], gold: list[int], k: int) -> float:
    """1.0 when any gold page appears in the top-k retrieved pages."""
    return 1.0 if rank_of_first_hit([(p, p) for p in pages], gold, k) else 0.0


def reciprocal_rank(pages: list[int], gold: list[int]) -> float:
    rank = rank_of_first_hit([(p, p) for p in pages], gold)
    return 1.0 / rank if rank else 0.0


def cited_numbers(answer: str) -> list[int]:
    """Citation ids in order of appearance, de-duplicated."""
    seen: list[int] = []
    for match in _CITE.finditer(answer or ""):
        number = int(match.group(1))
        if number not in seen:
            seen.append(number)
    return seen


def citation_grounding(answer: str, n_sources: int) -> tuple[int, int]:
    """(valid citations, total citations). A citation outside 1..n is a hallucination."""
    numbers = cited_numbers(answer)
    valid = sum(1 for n in numbers if 1 <= n <= n_sources)
    return valid, len(numbers)


def is_abstain(answer: str) -> bool:
    text = (answer or "").strip()
    if not text:
        return True
    return any(marker in text[:200] for marker in _ABSTAIN)


def summarize(rows: list[dict], k: int = 5) -> dict:
    """Aggregate per-question rows into the headline numbers."""
    answered = [row for row in rows if row.get("answerable", True)]
    abstainable = [row for row in rows if not row.get("answerable", True)]

    hit = sum(row["hit"] for row in answered) / len(answered) if answered else 0.0
    mrr = sum(row["rr"] for row in answered) / len(answered) if answered else 0.0

    valid = sum(row.get("cite_valid", 0) for row in rows)
    total = sum(row.get("cite_total", 0) for row in rows)
    grounding = valid / total if total else None

    abstain_ok = None
    if abstainable and any("abstained" in row for row in abstainable):
        abstain_ok = sum(1 for row in abstainable if row.get("abstained")) / len(abstainable)
    return {
        "k": k,
        "questions": len(rows),
        "answerable": len(answered),
        "hit": round(hit, 4),
        "mrr": round(mrr, 4),
        "grounding": None if grounding is None else round(grounding, 4),
        "abstain_accuracy": None if abstain_ok is None else round(abstain_ok, 4),
        "citation_total": total,
    }
