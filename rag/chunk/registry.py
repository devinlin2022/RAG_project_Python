"""Strategy dispatch and comparison."""

import time

from rag.chunk.hybrid import chunk_hybrid, chunk_structured
from rag.chunk.metrics import composite
from rag.chunk.profile import profile_document
from rag.chunk.recursive import chunk_recursive
from rag.chunk.semantic import chunk_semantic
from rag.chunk.types import Chunk

STRATEGIES = ("hybrid", "structured", "semantic", "recursive")

LABELS = {
    "hybrid": "混合（结构+语义）",
    "structured": "结构化（标题/条款）",
    "semantic": "语义切分",
    "recursive": "递归切分",
    "auto": "自动推荐",
}


def _as_text_pairs(pairs: list[tuple[int, object]]) -> list[tuple[int, str]]:
    return [(page, getattr(block, "text", "")) for page, block in pairs]


def _attach_pages(chunks: list[Chunk], pairs: list[tuple[int, object]]) -> list[Chunk]:
    """Re-derive page ranges from the source blocks.

    Splitters work on text, so a piece cut out of a 40-page section inherits the
    section's whole page span -- a citation then reads "p.1-46" and every page
    looks equally likely. Mapping each chunk back onto the block stream restores
    the real page, without threading page offsets through four splitters.
    """
    compact: list[str] = []
    page_of: list[int] = []
    for page, block in pairs:
        for char in getattr(block, "text", ""):
            if not char.isspace():
                compact.append(char)
                page_of.append(page)
    if not compact:
        return chunks

    source = "".join(compact)
    cursor = 0
    for chunk in chunks:
        flat = "".join(char for char in chunk.text if not char.isspace())
        if not flat:
            continue
        anchor = flat[:24]
        start = source.find(anchor, cursor)
        if start < 0:
            start = source.find(anchor)
        if start < 0:
            continue
        span = page_of[start : min(start + len(flat), len(page_of))]
        if span:
            chunk.page_from = min(span)
            chunk.page_to = max(span)
            cursor = start + 1
    return chunks


def _is_stub(chunk: Chunk) -> bool:
    """True for a chunk that holds titles and nothing else."""
    if chunk.kind != "text" or chunk.chars > 60:
        return False
    lines = [line.strip() for line in chunk.text.splitlines() if line.strip()]
    return bool(lines) and not any(line[-1] in "。；！？;!?" for line in lines)


def _drop_stubs(chunks: list[Chunk]) -> list[Chunk]:
    """Drop runs of title-only chunks: a table of contents becomes exactly that.

    Every entry of a 目录 page looks like a heading, so the splitter opens a
    section per entry and emits five chunks of pure titles. A lone short chunk
    may still be a real heading, so only runs of two or more are removed.
    """
    runs: list[list[int]] = []
    run: list[int] = []
    for index, chunk in enumerate(chunks):
        if _is_stub(chunk):
            run.append(index)
            continue
        if len(run) >= 2:
            runs.append(run)
        run = []
    if len(run) >= 2:
        runs.append(run)

    drop = {index for group in runs for index in group}
    if not drop:
        return chunks
    kept = [chunk for index, chunk in enumerate(chunks) if index not in drop]
    for index, chunk in enumerate(kept):
        chunk.idx = index
    return kept


def chunk_document(
    pairs: list[tuple[int, object]],
    strategy: str = "auto",
    size: int = 500,
    overlap: int = 80,
    min_chars: int = 120,
    percentile: float = 85.0,
    pages: int = 0,
    table_pages: int = 0,
    figure_pages: int = 0,
) -> dict:
    """Chunk a parsed document. `auto` profiles first, then dispatches."""
    if not pairs:
        return {"chunks": [], "strategy": strategy, "ms": 0, "info": {"error": "文档为空"}}

    profile = None
    if strategy == "auto":
        profile = profile_document(pairs, pages, table_pages, figure_pages)
        strategy = profile.strategy
        size = profile.chunk_size
        overlap = profile.overlap

    if strategy not in STRATEGIES:
        strategy = "recursive"

    started = time.perf_counter()
    info: dict = {}

    if strategy == "hybrid":
        chunks, info = chunk_hybrid(
            pairs, size=size, overlap=overlap, min_chars=min_chars, percentile=percentile
        )
    elif strategy == "structured":
        chunks, info = chunk_structured(
            pairs, size=size, overlap=overlap, min_chars=min_chars, strategy="structured"
        )
    elif strategy == "semantic":
        chunks, info = chunk_semantic(
            _as_text_pairs(pairs), size=size, overlap=overlap, min_chars=min_chars, percentile=percentile
        )
    else:
        chunks = chunk_recursive(_as_text_pairs(pairs), size=size, overlap=overlap)
        info = {"mode": "recursive"}

    elapsed = int((time.perf_counter() - started) * 1000)
    chunks = _drop_stubs(_attach_pages(chunks, pairs))
    source = "\n".join(getattr(block, "text", "") for _, block in pairs)
    typed = sum(1 for _, block in pairs if getattr(block, "type", "text") != "text")

    return {
        "chunks": chunks,
        "strategy": strategy,
        "ms": elapsed,
        "size": size,
        "overlap": overlap,
        "info": info,
        "metrics": composite(chunks, source, min_chars, size, typed),
        "profile": profile.to_dict() if profile else None,
    }


def compare_strategies(
    pairs: list[tuple[int, object]],
    size: int = 500,
    overlap: int = 80,
    min_chars: int = 120,
    percentile: float = 85.0,
    pages: int = 0,
    table_pages: int = 0,
    figure_pages: int = 0,
) -> list[dict]:
    """Run all four strategies and return comparable rows, recommendation first."""
    profile = profile_document(pairs, pages, table_pages, figure_pages)
    typed = sum(1 for _, block in pairs if getattr(block, "type", "text") != "text")
    rows: list[dict] = []

    for strategy in STRATEGIES:
        started = time.perf_counter()
        try:
            if strategy == "hybrid":
                chunks, _ = chunk_hybrid(
                    pairs, size=size, overlap=overlap, min_chars=min_chars, percentile=percentile
                )
            elif strategy == "structured":
                chunks, _ = chunk_structured(
                    pairs, size=size, overlap=overlap, min_chars=min_chars, strategy="structured"
                )
            elif strategy == "semantic":
                chunks, _ = chunk_semantic(
                    _as_text_pairs(pairs), size=size, overlap=overlap, min_chars=min_chars,
                    percentile=percentile,
                )
            else:
                chunks = chunk_recursive(_as_text_pairs(pairs), size=size, overlap=overlap)
            error = ""
        except Exception as exc:  # one bad strategy must not kill the table
            chunks, error = [], str(exc)

        chunks = _drop_stubs(_attach_pages(chunks, pairs))
        elapsed = int((time.perf_counter() - started) * 1000)
        source = "\n".join(getattr(block, "text", "") for _, block in pairs)
        metrics = composite(chunks, source, min_chars, size, typed) if chunks else {}
        rows.append(
            {
                "strategy": strategy,
                "label": LABELS[strategy],
                "recommended": strategy == profile.strategy,
                "chunks": len(chunks),
                "avg_chars": metrics.get("avg_chars", 0),
                "clause_integrity": metrics.get("clause_integrity", 0.0),
                "boundary_violation": metrics.get("boundary_violation", 0.0),
                "size_conformance": metrics.get("size_conformance", 0.0),
                "section_coverage": metrics.get("section_coverage", 0.0),
                "clause_applicable": metrics.get("clause_applicable", True),
                "boundary_applicable": metrics.get("boundary_applicable", True),
                "typed_preservation": metrics.get("typed_preservation"),
                "struct_score": metrics.get("struct_score", 0.0),
                "ms": elapsed,
                "error": error,
            }
        )

    rows.sort(key=lambda row: (not row["recommended"], -row["struct_score"]))
    return rows


def chunk_preview(chunk: Chunk, width: int = 90) -> str:
    text = " ".join(chunk.text.split())
    return text[:width] + ("…" if len(text) > width else "")
