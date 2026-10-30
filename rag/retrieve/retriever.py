"""Dense, lexical and fused retrieval. One generator, three strategies.

The rails are independent, so they run in two threads and the caller is told
about each stage as it lands -- that is what the UI's thinking panel renders.
`search()` consumes the same generator, so the trace can never drift from what
actually happened.
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Iterator

from rag.index.fusion import rrf
from rag.index.library import ChunkLibrary, Record
from rag.index.tokenize import tokenize


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


def _top_preview(library: ChunkLibrary, hits: list[tuple[int, float]], limit: int = 3) -> list[dict]:
    return [
        {
            "filename": library.records[row].filename,
            "page": library.records[row].page_from,
            "score": round(score, 4),
            "preview": " ".join(library.records[row].text.split())[:48],
        }
        for row, score in hits[:limit]
    ]


def search_steps(
    query: str,
    query_vec: list[float] | None,
    library: ChunkLibrary,
    strategy: str = "rrf",
    top_k: int = 6,
    candidates: int = 24,
    doc_ids: list[str] | None = None,
    rrf_k: int = 60,
) -> Iterator[tuple[str, dict]]:
    """Yield ("dense_start"|"dense_done"|..., payload) and finally ("final", ...).

    When the vector index is empty (no embedding key) a dense request degrades
    to sparse instead of returning nothing, and `info["degraded"]` says so.
    """
    info: dict = {"degraded": ""}
    pool = max(candidates, top_k)
    hits: dict[int, Hit] = {}
    timings: dict[str, float] = {}

    dense_order: list[int] = []
    sparse_order: list[int] = []
    want_dense = strategy in ("dense", "rrf") and query_vec is not None and len(library.dense)
    want_sparse = strategy in ("sparse", "rrf")

    if strategy == "dense" and not want_dense:
        info["degraded"] = "向量索引为空（未配置 Embedding 或未建索引），本次退化为关键词检索"
        strategy = "sparse"
        want_sparse = True

    def run_dense() -> tuple[list[tuple[int, float]], float]:
        started = time.perf_counter()
        found = library.dense_search(query_vec, pool, doc_ids)
        return found, round((time.perf_counter() - started) * 1000, 1)

    def run_sparse() -> tuple[list[tuple[int, float]], float]:
        started = time.perf_counter()
        found = library.sparse_search(query, pool, doc_ids)
        return found, round((time.perf_counter() - started) * 1000, 1)

    if want_dense:
        yield "dense_start", {"model": library.embed_model, "vectors": len(library.dense)}
    if want_sparse:
        yield "sparse_start", {"terms": tokenize(query)[:12], "chunks": len(library.records)}

    dense_result: list[tuple[int, float]] = []
    sparse_result: list[tuple[int, float]] = []
    if want_dense and want_sparse:
        with ThreadPoolExecutor(max_workers=2) as pool_exec:
            futures = {"dense": pool_exec.submit(run_dense), "sparse": pool_exec.submit(run_sparse)}
            for future in as_completed(list(futures.values())):
                name = next(key for key, value in futures.items() if value is future)
                found, ms = future.result()
                timings[f"{name}_ms"] = ms
                if name == "dense":
                    dense_result = found
                    for rank, (row, score) in enumerate(found, start=1):
                        hits[row] = Hit(library.records[row], score, dense=score, rank_dense=rank)
                    yield "dense_done", {
                        "hits": len(found),
                        "ms": ms,
                        "top": _top_preview(library, found),
                    }
                else:
                    sparse_result = found
                    for rank, (row, score) in enumerate(found, start=1):
                        if row in hits:
                            hits[row].sparse = score
                            hits[row].rank_sparse = rank
                        else:
                            hits[row] = Hit(library.records[row], score, sparse=score, rank_sparse=rank)
                    yield "sparse_done", {
                        "hits": len(found),
                        "ms": ms,
                        "top": _top_preview(library, found),
                    }
    else:
        if want_dense:
            dense_result, ms = run_dense()
            timings["dense_ms"] = ms
            for rank, (row, score) in enumerate(dense_result, start=1):
                hits[row] = Hit(library.records[row], score, dense=score, rank_dense=rank)
            yield "dense_done", {"hits": len(dense_result), "ms": ms, "top": _top_preview(library, dense_result)}
        if want_sparse:
            sparse_result, ms = run_sparse()
            timings["sparse_ms"] = ms
            for rank, (row, score) in enumerate(sparse_result, start=1):
                hits[row] = Hit(library.records[row], score, sparse=score, rank_sparse=rank)
            yield "sparse_done", {"hits": len(sparse_result), "ms": ms, "top": _top_preview(library, sparse_result)}

    dense_order = [row for row, _ in dense_result]
    sparse_order = [row for row, _ in sparse_result]

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

    if strategy == "rrf":
        yield "fuse_done", {
            "candidates": len(result),
            "ms": timings["fuse_ms"],
            "k": rrf_k,
            "dense": len(dense_order),
            "sparse": len(sparse_order),
        }
    yield "final", {"hits": result[:pool], "info": info}


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
    """Return ranked hits plus per-stage timings."""
    final: tuple[list[Hit], dict] = ([], {"timings": {}, "strategy": strategy, "degraded": ""})
    for stage, payload in search_steps(
        query, query_vec, library, strategy, top_k, candidates, doc_ids, rrf_k
    ):
        if stage == "final":
            final = (payload["hits"], payload["info"])
    return final


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
