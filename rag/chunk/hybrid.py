"""Structure-first chunking, with an optional semantic pass inside a section.

Never cut a clause; accumulate siblings up to the size budget; reset at the
top-level ancestor so a chunk cannot straddle two chapters.
"""

from rag.chunk.recursive import split_recursive
from rag.chunk.semantic import semantic_split
from rag.chunk.structure import build_sections
from rag.chunk.table import split_markdown_table_guard
from rag.embed.client import embed_ready
from rag.chunk.types import Chunk


def _top(path: str) -> str:
    return path.split(" > ")[0] if path else "（正文）"


def _span(first: str, last: str) -> str:
    if not first:
        return last or ""
    if not last or first == last:
        return first
    return f"{first} … {last}"


def chunk_structured(
    blocks: list[tuple[int, object]],
    size: int,
    overlap: int,
    min_chars: int,
    percentile: float = 85.0,
    semantic_inside: bool = False,
    strategy: str = "structured",
) -> tuple[list[Chunk], dict]:
    """Return chunks plus a small stats dict for the UI."""
    sections = build_sections(blocks)
    if not sections:
        return [], {"sections": 0, "oversized": 0, "semantic_inside": semantic_inside}

    # No embedding endpoint: skip the pass instead of failing once per section.
    if semantic_inside and not embed_ready():
        semantic_inside = False
        semantic_skipped = True
    else:
        semantic_skipped = False

    chunks: list[Chunk] = []
    oversized = 0
    semantic_used = 0

    buffer: list[str] = []
    first_label = ""
    last_label = ""
    group = ""
    level = 0
    page_from = 0
    page_to = 0
    merged = 0

    def flush() -> None:
        nonlocal buffer, first_label, last_label, group, level, page_from, page_to, merged
        if not buffer:
            return
        text = "\n".join(buffer).strip()
        buffer = []
        if not text:
            return
        chunks.append(
            Chunk(
                idx=len(chunks),
                text=text,
                page_from=page_from,
                page_to=page_to,
                kind="text",
                section=_span(first_label, last_label),
                level=level,
                strategy=strategy,
                merged_from=max(merged, 1),
            )
        )
        first_label = last_label = group = ""
        level = 0
        page_from = page_to = 0
        merged = 0

    for section in sections:
        if section.kind in ("table", "figure"):
            flush()
            pieces = (
                split_markdown_table_guard(section.text, size)
                if section.kind == "table"
                else [section.text]
            )
            for piece in pieces:
                chunks.append(
                    Chunk(
                        idx=len(chunks),
                        text=piece,
                        page_from=section.page_from,
                        page_to=section.page_to,
                        kind=section.kind,
                        section=section.path,
                        level=section.level,
                        strategy=strategy,
                    )
                )
            continue

        top = _top(section.path)
        if group and top != group:
            flush()

        if section.chars > size:
            # One clause longer than the budget: cut inside it, keep the path.
            flush()
            oversized += 1
            pieces: list[str]
            if semantic_inside:
                pieces, info = semantic_split(section.text, size, min_chars, percentile)
                if info.get("mode") == "percentile":
                    semantic_used += 1
            else:
                pieces = split_recursive(section.text, size, overlap)
            for piece in pieces:
                if not piece.strip():
                    continue
                chunks.append(
                    Chunk(
                        idx=len(chunks),
                        text=piece.strip(),
                        page_from=section.page_from,
                        page_to=section.page_to,
                        kind="text",
                        section=section.path,
                        level=section.level,
                        strategy=strategy,
                    )
                )
            continue

        current = len("\n".join(buffer))
        if buffer and current + len(section.text) + 1 > size:
            flush()

        if not buffer:
            group = top
            level = section.level
            first_label = section.path
            page_from = section.page_from
            page_to = section.page_to
            merged = 1
        else:
            page_to = max(page_to, section.page_to)
            merged += 1
        last_label = section.path
        buffer.append(section.text)

    flush()

    # Merge a short tail only within the same top-level section.
    if len(chunks) >= 2:
        tail = chunks[-1]
        head = chunks[-2]
        if (
            tail.kind == "text"
            and head.kind == "text"
            and _top(tail.section) == _top(head.section)
            and tail.chars < min_chars
            and head.chars + tail.chars + 1 <= size
        ):
            head.text = head.text.rstrip() + "\n" + tail.text.lstrip()
            head.page_to = max(head.page_to, tail.page_to)
            head.section = _span(head.section.split(" … ")[0], tail.section.split(" … ")[-1])
            chunks.pop()
            for index, chunk in enumerate(chunks):
                chunk.idx = index

    for index, chunk in enumerate(chunks):
        chunk.idx = index

    stats = {
        "sections": len(sections),
        "oversized": oversized,
        "semantic_inside": semantic_inside,
        "semantic_sections": semantic_used,
        "semantic_skipped_no_embed": semantic_skipped,
    }
    return chunks, stats


def chunk_hybrid(
    blocks: list[tuple[int, object]],
    size: int,
    overlap: int,
    min_chars: int,
    percentile: float = 85.0,
) -> tuple[list[Chunk], dict]:
    """Structure first, semantic inside oversized sections, tables kept as tables."""
    return chunk_structured(
        blocks,
        size=size,
        overlap=overlap,
        min_chars=min_chars,
        percentile=percentile,
        semantic_inside=True,
        strategy="hybrid",
    )
