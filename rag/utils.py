"""Dependency-free helpers, testable without installing anything."""

import re

_PARTS = re.compile(r"[,，;；\s]+")
_RANGE = re.compile(r"^(\d+)\s*[-~]\s*(\d+)$")


def parse_pages(expr: str, limit: int = 5000) -> list[int]:
    """Parse "1-5,8" into a sorted page list.

    >>> parse_pages("1-5,8")
    [1, 2, 3, 4, 5, 8]
    """
    if not expr or not expr.strip():
        return []

    pages: set[int] = set()
    for part in _PARTS.split(expr.strip()):
        if not part:
            continue
        matched = _RANGE.match(part)
        if matched:
            start, end = int(matched.group(1)), int(matched.group(2))
            if start > end:
                start, end = end, start
            if end - start > limit:
                end = start + limit
            pages.update(range(start, end + 1))
        elif part.isdigit():
            pages.add(int(part))
    return sorted(p for p in pages if p >= 1)


def format_pages(pages: list[int], limit: int = 40) -> str:
    """Compress pages into ranges: [1,2,3,7] -> "1-3, 7"."""
    if not pages:
        return "-"

    ordered = sorted(set(pages))
    if len(ordered) > limit:
        return f"{ordered[0]}-{ordered[-1]}（共 {len(ordered)} 页）"

    spans: list[tuple[int, int]] = []
    start = prev = ordered[0]
    for page in ordered[1:]:
        if page == prev + 1:
            prev = page
            continue
        spans.append((start, prev))
        start = prev = page
    spans.append((start, prev))

    return ", ".join(f"{a}-{b}" if a != b else str(a) for a, b in spans)
