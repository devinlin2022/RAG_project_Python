"""Dense, lexical and fused retrieval. One function, three strategies."""

import time
from dataclasses import dataclass

from rag.index.fusion import rrf
from rag.index.library import ChunkLibrary, Record


@dataclass
class Hit:
    record: Record
    score: float
    dense: float | None = None
    sparse: float | None = None
    rerank: float | None = None
    rank_dense: int | None = None
    rank_sparse: int | None = None

    @property
    def page(self) -> str:
        return self.record.page_label


def search(
    query: str,
    query_vec: list[float] | None,
    library: ChunkLibrary,
    strategy: str = "rrf",
    top_k: int = 6,
    candidates: int = 24,
    doc_ids: list[str] | None = None,
    rrf_k: int = 60,
) -> tuple[list[Hit], dict]:
    """Return ranked hits plus per-stage timings.

    `strategy` is dense / sparse / rrf. When the vector index is empty (no
    embedding key) a dense request degrades to sparse instead of returning
    nothing, and the caller is told through `info["degraded"]`.
    """
    info: dict = {"degraded": ""}
    pool = max(candidates, top_k)
    hits: dict[int, Hit] = {}
    timings: dict[str, float] = {}

    dense_order: list[int] = []
    sparse_order: list[int] = []

    started = time.perf_counter()
    if strategy in ("dense", "rrf") and query_vec is not None and len(library.dense):
        for rank, (row, score) in enumerate(
            library.dense_search(query_vec, pool, doc_ids), start=1
        ):
            record = library.records[row]
            hits[row] = Hit(record=record, score=score, dense=score, rank_dense=rank)
            dense_order.append(row)
    elif strategy == "dense":
        info["degraded"] = "向量索引为空（未配置 Embedding 或未建索引），本次退化为关键词检索"
        strategy = "sparse"
    timings["dense_ms"] = round((time.perf_counter() - started) * 1000, 1)

    started = time.perf_counter()
    if strategy in ("sparse", "rrf"):
        for rank, (row, score) in enumerate(library.sparse_search(query, pool, doc_ids), start=1):
            if row in hits:
                hits[row].sparse = score
                hits[row].rank_sparse = rank
            else:
                hits[row] = Hit(record=library.records[row], score=score, sparse=score, rank_sparse=rank)
            sparse_order.append(row)
    timings["sparse_ms"] = round((time.perf_counter() - started) * 1000, 1)

    started = time.perf_counter()
    if strategy == "rrf":
        fused = rrf([dense_order, sparse_order], k=rrf_k)
        for row, score in fused.items():
            hits[row].score = score
        ordered = sorted(fused.items(), key=lambda item: -item[1])
        result = [hits[row] for row, _ in ordered]
    else:
        result = sorted(hits.values(), key=lambda hit: -hit.score)
    timings["fuse_ms"] = round((time.perf_counter() - started) * 1000, 1)

    info["strategy"] = strategy
    info["dense_hits"] = len(dense_order)
    info["sparse_hits"] = len(sparse_order)
    info["timings"] = timings
    return result[:pool], info


def diversify(hits: list[Hit], top_k: int) -> list[Hit]:
    """Keep one chunk per page, so every citation slot adds new evidence.

    A paragraph split across two chunks can occupy two of six slots and cite the
    same page twice. The dropped duplicates are replaced from deeper ranks, which
    is why this runs before the final truncation rather than after.
    """
    seen: set[tuple[str, int]] = set()
    kept: list[Hit] = []
    for hit in hits:
        key = (hit.record.doc_id, hit.record.page_from)
        if key in seen:
            continue
        seen.add(key)
        kept.append(hit)
        if len(kept) >= top_k:
            break
    return kept


def rerank_hits(
    query: str,
    hits: list[Hit],
    top_k: int,
    reranker=None,
) -> tuple[list[Hit], dict]:
    """Cross-encoder rerank in place; keeps fusion order when unavailable."""
    info = {"used": False, "model": "", "ms": 0.0, "error": ""}
    if not hits:
        return hits, info
    if reranker is None:
        return hits[:top_k], info

    started = time.perf_counter()
    try:
        scored = reranker(query, [hit.record.text for hit in hits])
    except Exception as exc:  # a broken reranker must not break answering
        info["error"] = str(exc)
        info["ms"] = round((time.perf_counter() - started) * 1000, 1)
        return hits[:top_k], info

    for index, score in scored:
        if 0 <= index < len(hits):
            hits[index].rerank = score
    ranked = sorted(
        (hit for hit in hits if hit.rerank is not None), key=lambda hit: -hit.rerank
    )
    info.update(
        used=True,
        ms=round((time.perf_counter() - started) * 1000, 1),
    )
    return ranked[:top_k], info
