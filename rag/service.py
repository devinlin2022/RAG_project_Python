"""Orchestration layer: ingest, index, retrieve, answer, evaluate.

The UI and the HTTP API both call this module, so one process keeps one index
state. The answering path is a generator, which lets the UI render each stage
as it happens.
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from rag.chunk.registry import chunk_document
from rag.config import VERSION, settings
from rag.embed import client as embed_client
from rag.embed.client import EmbedError, embed_texts
from rag.generate.llm import LLMError, chat, chat_stream
from rag.generate.prompt import NO_CONTEXT_ANSWER, build_chunk_messages
from rag.generate.rewrite import needs_rewrite, rewrite_question
from rag.index.library import ChunkLibrary
from rag.ingest.enhance import Block, all_blocks, enhance
from rag.ingest.loader import load_document, sha256_file
from rag.retrieve import reranker
from rag.retrieve.retriever import Hit, diversify, rerank_hits, search, search_steps
from rag.store.db import Database

log = logging.getLogger(__name__)

STRATEGY_LABELS = {
    "rrf": "RRF 融合",
    "dense": "向量",
    "sparse": "BM25",
}

# Two user-facing modes, so nobody has to understand RRF before asking a question.
# `fast` is the cheap default; `enhanced` pays for rewrite + rerank + a wider pool.
MODES = {
    "fast": {
        "label": "快速检索",
        "strategy": "rrf",
        "top_k": 5,
        "candidates": 18,
        "rerank": False,
        "rewrite": False,
    },
    "enhanced": {
        "label": "增强检索",
        "strategy": "rrf",
        "top_k": 8,
        "candidates": 36,
        "rerank": True,
        "rewrite": True,
    },
}


class ServiceError(ValueError):
    """Invalid request; the message is safe to show."""


def _ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 1)


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _norm(vector) -> float:
    if vector is None:
        return 0.0
    total = sum(float(value) * float(value) for value in vector) ** 0.5
    return round(total, 3)


class Service:
    def __init__(self, db: Database | None = None) -> None:
        self.db = db or Database(settings.db_path)
        self.library = ChunkLibrary(self.db)
        self.docs: dict[str, dict] = {}
        self.started_at = _now()
        self.usage = self.db.usage_totals()
        self._load_docs()

    # ---------- documents ----------

    def _load_docs(self) -> None:
        for meta in self.db.list_docs():
            pages = self.db.load_pages(meta["id"])
            if not pages:
                continue
            blocks = self.db.load_blocks(meta["id"])
            self.docs[meta["id"]] = {
                "id": meta["id"],
                "filename": meta["filename"],
                "path": meta["path"],
                "format": meta.get("format") or "",
                "pages": pages,
                "page_count": len(pages),
                "chars": sum(p["chars"] for p in pages),
                "scan_like_pages": [p["page"] for p in pages if p["scan_like"]],
                "table_like_pages": [p["page"] for p in pages if p["table_like"]],
                "figure_like_pages": [p["page"] for p in pages if p["figure_like"]],
                "created_at": meta["created_at"],
                "strategy": meta["strategy"],
                "chunk_count": meta["chunk_count"],
                # dim is the only proof that this document has vectors at all.
                "dim": meta.get("dim", 0),
                "indexed_at": meta["indexed_at"],
                "enhanced_at": meta["enhanced_at"],
                "blocks": {
                    page: [Block(item["type"], item["text"], item["source"]) for item in items]
                    for page, items in blocks.items()
                },
            }
        if self.docs:
            log.info("restored %d documents from %s", len(self.docs), self.db.path)

    def _meta(self, doc: dict) -> dict:
        return {
            "id": doc["id"],
            "filename": doc["filename"],
            "path": doc["path"],
            "format": doc.get("format") or "",
            "pages": doc["page_count"],
            "chars": doc["chars"],
            "created_at": doc.get("created_at") or _now(),
            "strategy": doc.get("strategy"),
            "chunk_count": doc.get("chunk_count", 0),
            "dim": doc.get("dim", 0),
            "indexed_at": doc.get("indexed_at"),
            "enhanced_at": doc.get("enhanced_at"),
        }

    def ingest(self, path: str | Path, filename: str | None = None) -> tuple[dict, bool]:
        """Parse and persist one file. Returns (doc, was_already_present)."""
        path = Path(path)
        doc_id = sha256_file(path)[:16]
        if doc_id in self.docs:
            return self.docs[doc_id], True

        doc = load_document(path, doc_id=doc_id, filename=filename)
        doc["dim"] = 0
        self.db.upsert_doc(self._meta(doc))
        self.db.save_pages(doc_id, doc["pages"])
        self.docs[doc_id] = doc
        log.info(
            "ingest doc=%s format=%s pages=%d chars=%d",
            doc_id,
            doc.get("format", ""),
            doc["page_count"],
            doc["chars"],
        )
        return doc, False

    def ingest_files(self, paths: list[str], on_event=None) -> list[dict]:
        """Parse several uploads in parallel, then report in input order."""
        emit = on_event or (lambda event: None)
        results: list[dict] = [{} for _ in paths]

        def task(index: int, path: str) -> None:
            name = Path(path).name
            started = time.perf_counter()
            try:
                doc, existed = self.ingest(path, name)
                results[index] = {
                    "ok": True,
                    "existed": existed,
                    "doc": doc,
                    "filename": name,
                    "ms": int(_ms(started)),
                }
            except Exception as exc:  # one bad file must not stop the batch
                results[index] = {"ok": False, "filename": name, "error": str(exc), "ms": int(_ms(started))}
            emit({"stage": "file_done", "filename": name, "ok": results[index]["ok"]})

        workers = max(1, min(int(settings.parse_workers), len(paths) or 1))
        if workers == 1:
            for index, path in enumerate(paths):
                emit({"stage": "file_start", "filename": Path(path).name})
                task(index, path)
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {}
                for index, path in enumerate(paths):
                    emit({"stage": "file_start", "filename": Path(path).name})
                    futures[pool.submit(task, index, path)] = index
                for future in futures:
                    future.result()
        return results

    def get(self, doc_id: str | None) -> dict:
        if not doc_id or doc_id not in self.docs:
            raise ServiceError("文档不存在，请先上传并解析文件")
        return self.docs[doc_id]

    def list_documents(self) -> list[dict]:
        rows = []
        for doc in self.docs.values():
            rows.append(
                {
                    "id": doc["id"],
                    "filename": doc["filename"],
                    "format": doc.get("format") or "—",
                    "pages": doc["page_count"],
                    "chars": doc["chars"],
                    "chunks": doc.get("chunk_count", 0),
                    "dim": doc.get("dim", 0),
                    "strategy": doc.get("strategy") or "—",
                    "indexed_at": doc.get("indexed_at") or "—",
                    "enhanced": bool(doc.get("enhanced_at")),
                    "created_at": doc.get("created_at", ""),
                }
            )
        return sorted(rows, key=lambda row: row["created_at"])

    def delete(self, doc_id: str) -> bool:
        doc = self.docs.pop(doc_id, None)
        if doc is None:
            return False
        self.library.remove_document(doc_id)
        path = Path(doc["path"])
        # Only ever delete the uploaded copy: a document parsed in place from the
        # user's own folder must not disappear because its index was dropped.
        if path.is_relative_to(settings.upload_dir):
            try:
                path.unlink(missing_ok=True)
            except OSError as exc:  # a locked file must not block the delete
                log.warning("could not remove %s: %s", path, exc)
        return True

    def pages(self, doc_id: str) -> list[dict]:
        return self.get(doc_id)["pages"]

    def doc_ids_matching(self, pattern: str) -> list[str]:
        needle = (pattern or "").strip().lower()
        if not needle:
            return list(self.docs)
        return [d["id"] for d in self.docs.values() if needle in d["filename"].lower()]

    # ---------- enhancement ----------

    def enhance(
        self, doc_id: str, limit_pages: int | None = None, on_event=None, force: bool = False
    ) -> dict:
        doc = self.get(doc_id)
        report = enhance(doc, limit_pages=limit_pages, on_event=on_event, force=force)
        self.db.save_blocks(doc_id, doc.get("blocks", {}))
        doc["enhanced_at"] = _now()
        self.db.mark_enhanced(doc_id)
        return report

    # ---------- indexing ----------

    def index(
        self,
        doc_id: str,
        strategy: str = "auto",
        size: int | None = None,
        overlap: int | None = None,
        force: bool = False,
    ) -> dict:
        doc = self.get(doc_id)
        # A document indexed while embedding was down has no vectors; index it again
        # now that the key works instead of skipping it forever.
        needs_vectors = settings.embed_ready and not doc.get("dim")
        if doc.get("chunk_count") and doc.get("indexed_at") and not force and not needs_vectors:
            return {
                "doc_id": doc_id,
                "chunks": doc["chunk_count"],
                "dim": doc.get("dim", 0),
                "strategy": doc.get("strategy"),
                "skipped": True,
                "embed_ms": 0,
                "embed_error": "",
            }

        started = time.perf_counter()
        result = chunk_document(
            all_blocks(doc),
            strategy=strategy,
            size=size or settings.chunk_size,
            overlap=settings.chunk_overlap if overlap is None else overlap,
            min_chars=settings.chunk_min_chars,
            percentile=settings.semantic_percentile,
            pages=doc["page_count"],
            table_pages=len(doc["table_like_pages"]),
            figure_pages=len(doc["figure_like_pages"]),
            slide_deck=(doc.get("format") == "PPT"),
        )
        chunks = result["chunks"]
        if not chunks:
            raise ServiceError("切分结果为空，无法建立索引")

        vectors: list[list[float]] | None = None
        embed_ms = 0.0
        embed_error = ""
        if settings.embed_ready:
            embed_started = time.perf_counter()
            try:
                vectors = embed_texts([chunk.text for chunk in chunks])
            except EmbedError as exc:
                embed_error = str(exc)
            embed_ms = _ms(embed_started)

        payload = []
        for position, chunk in enumerate(chunks):
            item = chunk.to_dict()
            item["vec"] = vectors[position] if vectors else None
            payload.append(item)
        self.library.replace_document(doc_id, payload)

        dim = len(vectors[0]) if vectors else 0
        doc["strategy"] = result["strategy"]
        doc["chunk_count"] = len(chunks)
        doc["dim"] = dim
        doc["indexed_at"] = _now()
        self.db.upsert_doc(self._meta(doc))
        if vectors:
            self._add_usage(embed_tokens=embed_client.last_usage.get("prompt_tokens", 0))

        return {
            "doc_id": doc_id,
            "chunks": len(chunks),
            "dim": dim,
            "strategy": result["strategy"],
            "skipped": False,
            "embed_ms": embed_ms,
            "embed_error": embed_error,
            "ms": _ms(started),
            "metrics": result.get("metrics", {}),
        }

    def index_all(self, force: bool = False) -> list[dict]:
        return [self.index(doc_id, force=force) for doc_id in list(self.docs)]

    # ---------- usage ----------

    def _add_usage(self, llm_in: int = 0, llm_out: int = 0, embed_tokens: int = 0) -> None:
        self.usage["llm_in"] = self.usage.get("llm_in", 0) + int(llm_in or 0)
        self.usage["llm_out"] = self.usage.get("llm_out", 0) + int(llm_out or 0)
        self.usage["embed"] = self.usage.get("embed", 0) + int(embed_tokens or 0)
        self.usage["requests"] = self.usage.get("requests", 0) + 1

    def usage_summary(self) -> dict:
        total = self.usage.get("llm_in", 0) + self.usage.get("llm_out", 0)
        cost = (
            self.usage.get("llm_in", 0) / 1_000_000 * settings.price_in_per_m
            + self.usage.get("llm_out", 0) / 1_000_000 * settings.price_out_per_m
        )
        return {
            **self.usage,
            "total_tokens": total,
            "cost": round(cost, 4),
            "priced": bool(settings.price_in_per_m or settings.price_out_per_m),
        }

    # ---------- retrieval ----------

    def _embed_query(self, query: str) -> tuple[list[float] | None, float, str]:
        if not settings.embed_ready or not len(self.library.dense):
            return None, 0.0, ""
        started = time.perf_counter()
        try:
            return embed_texts([query])[0], _ms(started), ""
        except EmbedError as exc:
            return None, _ms(started), str(exc)

    def _reranker(self):
        return reranker.rerank if settings.rerank_ready else None

    def retrieve(
        self,
        query: str,
        doc_ids: list[str] | None = None,
        top_k: int | None = None,
        strategy: str | None = None,
        rerank: bool | None = None,
        candidates: int | None = None,
    ) -> dict:
        """Retrieve + optional rerank. Shared by ask, debug and evaluation."""
        if not query or not query.strip():
            raise ServiceError("问题不能为空")
        top_k = top_k or settings.retrieve_top_k
        strategy = strategy or settings.retrieve_strategy
        if strategy not in ("dense", "sparse", "rrf"):
            raise ServiceError(f"未知检索策略：{strategy}")

        started = time.perf_counter()
        vector, embed_ms, embed_error = self._embed_query(query)
        hits, info = search(
            query,
            vector,
            self.library,
            strategy=strategy,
            top_k=top_k,
            candidates=candidates or settings.rerank_candidates,
            doc_ids=doc_ids,
            rrf_k=settings.rrf_k,
        )
        candidates = len(hits)

        want_rerank = settings.rerank_ready if rerank is None else (rerank and settings.rerank_ready)
        rerank_info = {"used": False, "model": "", "ms": 0.0, "error": ""}
        if want_rerank and hits:
            # Rerank the whole candidate pool, then dedupe, then cut: slicing
            # first would waste slots on a second chunk of the same page.
            hits, rerank_info = rerank_hits(query, hits, len(hits), reranker.rerank)
            rerank_info["model"] = settings.rerank_model
        hits = diversify(hits, top_k)

        return {
            "hits": hits,
            "query": query,
            "strategy": info["strategy"],
            "candidates": candidates,
            "timings": {
                "embed_ms": embed_ms,
                "retrieve_ms": _ms(started),
                **info["timings"],
                "rerank_ms": rerank_info["ms"],
            },
            "rerank": rerank_info,
            "degraded": info.get("degraded", ""),
            "embed_error": embed_error,
            "doc_ids": doc_ids,
        }

    def retrieve_mode(
        self,
        query: str,
        doc_ids: list[str] | None = None,
        mode: str = "fast",
        top_k: int = 5,
    ) -> dict:
        """One question through one mode's settings, without generating an answer."""
        plan = self.resolve_mode(mode)
        result = self.retrieve(
            query,
            doc_ids,
            top_k=top_k,
            strategy=plan["strategy"],
            rerank=plan["rerank"],
            candidates=max(int(plan["candidates"]), top_k),
        )
        result["mode"] = plan
        return result

    def compare_modes(
        self, query: str, doc_ids: list[str] | None = None, top_k: int = 5
    ) -> dict:
        """Run the same question through both modes and compare the retrieved pages.

        Both runs stop before generation, so this costs one embedding and, in
        enhanced mode, one rerank.
        """
        rows: list[dict] = []
        for name in ("fast", "enhanced"):
            result = self.retrieve_mode(query, doc_ids, mode=name, top_k=top_k)
            hits = result["hits"]
            pages: list[int] = []
            for hit in hits:
                for page in range(hit.record.page_from, hit.record.page_to + 1):
                    if page not in pages:
                        pages.append(page)
            scores = [hit.score for hit in hits]
            head = scores[0] if scores else 0.0
            rest = scores[1:] or [0.0]
            total_pages = self._scope_pages(doc_ids)
            rows.append(
                {
                    "mode": name,
                    "label": MODES[name]["label"],
                    "hits": len(hits),
                    "pages": sorted(pages),
                    "coverage": round(len(pages) / total_pages, 4) if total_pages else 0.0,
                    "top_score": round(head, 6),
                    "margin": round(head - sum(rest) / len(rest), 6),
                    "rerank_used": bool(result["rerank"].get("used")),
                    "rerank_top": round(
                        max((hit.rerank or 0.0) for hit in hits), 6
                    ) if hits else 0.0,
                    "ms": round(result["timings"].get("retrieve_ms", 0.0)),
                    "keys": {f"{hit.record.doc_id}:{hit.record.page_from}" for hit in hits},
                    "pages_set": set(pages),
                }
            )

        fast, enhanced = rows
        new_pages = sorted(enhanced["pages_set"] - fast["pages_set"])
        dropped = sorted(fast["pages_set"] - enhanced["pages_set"])
        gained = len(new_pages)

        if enhanced["pages_set"] == fast["pages_set"]:
            advice = "两种模式命中的是同一批页，这个问题用「快速检索」就够，省一次改写和一次精排。"
        elif gained and not dropped:
            listed = "、".join(f"p.{page}" for page in new_pages[:4])
            advice = (
                f"「增强检索」在同样的问题上多带回了 {gained} 页（{listed}），"
                "问题涉及跨段信息时值得多花这点时间。"
            )
        elif gained and dropped:
            listed_in = "、".join(f"p.{page}" for page in new_pages[:3])
            listed_out = "、".join(f"p.{page}" for page in dropped[:3])
            advice = (
                f"「增强检索」换入了 {listed_in}，挤出了 {listed_out}。"
                "静默改写会扩大召回面，也会挤掉原本靠前的块——答案出现偏移时先看这里。"
            )
        else:
            advice = "「增强检索」没有新增页，还挤掉了几页，本次用「快速检索」更稳妥。"

        return {
            "query": query,
            "top_k": top_k,
            "rows": [{key: value for key, value in row.items() if key not in ("keys", "pages_set")} for row in rows],
            "new_pages": new_pages,
            "dropped_pages": dropped,
            "advice": advice,
        }

    def _scope_pages(self, doc_ids: list[str] | None) -> int:
        docs = [
            doc for doc_id, doc in self.docs.items() if not doc_ids or doc_id in doc_ids
        ]
        return sum(doc["page_count"] for doc in docs) or 0

    def debug(self, query: str, doc_ids: list[str] | None = None, top_k: int = 5) -> dict:
        """Run all three strategies on one query for the comparison table."""
        rows: dict[str, list[dict]] = {}
        timings: dict[str, float] = {}
        for strategy in ("dense", "sparse", "rrf"):
            result = self.retrieve(query, doc_ids, top_k=top_k, strategy=strategy, rerank=False)
            rows[strategy] = [_hit_dict(hit, number) for number, hit in enumerate(result["hits"], 1)]
            timings[strategy] = result["timings"]["retrieve_ms"]

        fused = self.retrieve(query, doc_ids, top_k=top_k, strategy="rrf", rerank=True)
        return {
            "query": query,
            "rails": rows,
            "timings": timings,
            "reranked": [_hit_dict(hit, number) for number, hit in enumerate(fused["hits"], 1)],
            "rerank": fused["rerank"],
            "degraded": fused["degraded"],
        }

    # ---------- answering ----------

    def resolve_mode(self, mode: str | None) -> dict:
        if not mode or mode not in MODES:
            return {
                "name": "custom",
                "label": "自定义",
                "strategy": settings.retrieve_strategy,
                "top_k": settings.retrieve_top_k,
                "candidates": settings.rerank_candidates,
                "rerank": settings.rerank_enabled,
                "rewrite": settings.rewrite_enabled,
            }
        return {"name": mode, **MODES[mode]}

    def prepare_steps(
        self,
        question: str,
        doc_ids: list[str] | None = None,
        top_k: int | None = None,
        strategy: str | None = None,
        rerank: bool | None = None,
        history: list[dict] | None = None,
        mode: str | None = None,
        rewrite: bool | None = None,
    ):
        """Yield pipeline events, then ("prepared", payload).

        Every event carries the numbers the UI renders: dimension, norm, hit count,
        elapsed time.
        """
        plan = self.resolve_mode(mode)
        if not (question or "").strip():
            raise ServiceError("问题不能为空")
        top_k = top_k or plan["top_k"]
        strategy = strategy or plan["strategy"]
        # A rewrite is only worth a round trip when the question points back at
        # something like "what about that one"; without a key the helper no-ops.
        want_rewrite = plan["rewrite"] if rewrite is None else rewrite
        want_rerank = plan["rerank"] if rerank is None else rerank

        started_all = time.perf_counter()

        # 1. rewrite
        query = question.strip()
        rewritten, was_rewritten, rewrite_error = query, False, ""
        if want_rewrite and needs_rewrite(query, history):
            yield {"stage": "rewrite_start", "question": query}
            rewrite_started = time.perf_counter()
            rewritten, was_rewritten, rewrite_error = rewrite_question(
                question,
                history,
                chat,
                turns=settings.history_turns,
                enabled=settings.llm_configured,
            )
            yield {
                "stage": "rewrite_done",
                "query": rewritten,
                "rewritten": was_rewritten,
                "ms": _ms(rewrite_started),
                "error": rewrite_error,
                "skipped": not settings.llm_configured,
            }
        rewrite_ms = _ms(started_all) if want_rewrite else 0.0

        # 2. embed the query
        yield {
            "stage": "embed_start",
            "model": settings.embed_model,
            "ready": settings.embed_ready and bool(len(self.library.dense)),
        }
        vector, embed_ms, embed_error = self._embed_query(rewritten)
        if embed_error:
            yield {"stage": "embed_done", "error": embed_error, "ms": embed_ms}
        else:
            yield {
                "stage": "embed_done",
                "ms": embed_ms,
                "dim": len(vector) if vector else 0,
                "norm": _norm(vector),
                "tokens": embed_client.last_usage.get("prompt_tokens", 0),
                "skipped": vector is None,
            }

        # 3. the two rails, then fusion
        hits: list[Hit] = []
        info: dict = {"timings": {}, "strategy": strategy, "degraded": ""}
        for stage, payload in search_steps(
            rewritten,
            vector,
            self.library,
            strategy=strategy,
            top_k=top_k,
            candidates=max(plan["candidates"], top_k),
            doc_ids=doc_ids,
            rrf_k=settings.rrf_k,
        ):
            if stage == "final":
                hits, info = payload["hits"], payload["info"]
                continue
            yield {"stage": stage, **payload}

        candidates = len(hits)

        # 4. rerank
        rerank_info = {"used": False, "model": "", "ms": 0.0, "error": ""}
        if want_rerank and settings.rerank_ready and hits:
            yield {"stage": "rerank_start", "candidates": len(hits), "model": settings.rerank_model}
            hits, rerank_info = rerank_hits(rewritten, hits, len(hits), reranker.rerank)
            rerank_info["model"] = settings.rerank_model
            yield {
                "stage": "rerank_done",
                "ms": rerank_info["ms"],
                "used": rerank_info["used"],
                "error": rerank_info["error"],
                "top": [
                    {
                        "filename": hit.record.filename,
                        "page": hit.record.page_from,
                        "score": round(hit.rerank or 0, 3),
                    }
                    for hit in hits[:3]
                ],
            }
        elif want_rerank and not settings.rerank_ready:
            yield {"stage": "rerank_skipped", "reason": "未配置 Key 或已关闭精排"}

        hits = diversify(hits, top_k)
        retrieve_ms = _ms(started_all)

        payload = {
            "question": question.strip(),
            "query": rewritten,
            "rewritten": was_rewritten,
            "rewrite_error": rewrite_error,
            "hits": hits,
            "timings": {
                "rewrite_ms": rewrite_ms,
                "embed_ms": embed_ms,
                "retrieve_ms": retrieve_ms,
                **info["timings"],
                "rerank_ms": rerank_info["ms"],
            },
            "rerank": rerank_info,
            "degraded": info.get("degraded", ""),
            "embed_error": embed_error,
            "candidates": candidates,
            "strategy": info["strategy"],
            "mode": plan,
        }
        yield {
            "stage": "retrieved",
            "sources": [_source_dict(hit, number) for number, hit in enumerate(hits, 1)],
            "timings": payload["timings"],
            "candidates": candidates,
            "strategy": info["strategy"],
            "degraded": payload["degraded"],
            "rerank": rerank_info,
        }
        yield "prepared", payload

    def _prepare(
        self,
        question: str,
        doc_ids: list[str] | None,
        top_k: int | None,
        strategy: str | None,
        rerank: bool | None,
        history: list[dict] | None,
        mode: str | None = None,
        rewrite: bool | None = None,
    ) -> dict:
        prepared: dict = {}
        for event in self.prepare_steps(
            question, doc_ids, top_k, strategy, rerank, history, mode, rewrite
        ):
            if isinstance(event, tuple) and event[0] == "prepared":
                prepared = event[1]
        return prepared

    def _finish(self, prepared: dict, answer: str, usage: dict, generate_ms: float) -> dict:
        hits = prepared["hits"]
        from eval.metrics import citation_grounding, is_abstain

        valid, total = citation_grounding(answer, len(hits))
        tokens_in = int((usage or {}).get("prompt_tokens") or 0)
        tokens_out = int((usage or {}).get("completion_tokens") or 0)
        payload = {
            "answer": answer,
            "question": prepared["question"],
            "query": prepared["query"],
            "rewritten": prepared["rewritten"],
            "rewrite_error": prepared["rewrite_error"],
            "sources": [_source_dict(hit, number) for number, hit in enumerate(hits, 1)],
            "timings": {
                **prepared["timings"],
                "generate_ms": round(generate_ms, 1),
                "total_ms": round(
                    prepared["timings"]["rewrite_ms"]
                    + prepared["timings"]["retrieve_ms"]
                    + generate_ms,
                    1,
                ),
            },
            "strategy": prepared["strategy"],
            "mode": prepared.get("mode", {}),
            "candidates": prepared.get("candidates", len(hits)),
            "rerank": prepared["rerank"],
            "degraded": prepared["degraded"],
            "embed_error": prepared["embed_error"],
            "abstained": is_abstain(answer),
            "citations": {"valid": valid, "total": total},
            "usage": {
                "prompt_tokens": tokens_in,
                "completion_tokens": tokens_out,
                "total_tokens": tokens_in + tokens_out,
            },
            "model": settings.llm_model,
        }
        self._add_usage(llm_in=tokens_in, llm_out=tokens_out)
        payload["usage"]["session"] = self.usage_summary()
        self.db.log_query(
            question=prepared["question"],
            answer=answer,
            pages=sorted({hit.record.page_from for hit in hits}),
            ms=int(payload["timings"]["total_ms"]),
            top_k=len(hits),
            strategy=prepared["strategy"],
            rewritten=prepared["query"] if prepared["rewritten"] else "",
            reranked=bool(prepared["rerank"]["used"]),
            tokens_in=tokens_in,
            tokens_out=tokens_out,
        )
        return payload

    def ask(
        self,
        question: str,
        doc_ids: list[str] | None = None,
        top_k: int | None = None,
        strategy: str | None = None,
        rerank: bool | None = None,
        history: list[dict] | None = None,
        mode: str | None = None,
    ) -> dict:
        prepared = self._prepare(question, doc_ids, top_k, strategy, rerank, history, mode)
        started = time.perf_counter()

        if not prepared["hits"]:
            return self._finish(prepared, NO_CONTEXT_ANSWER, {}, _ms(started))

        messages = build_chunk_messages(
            prepared["hits"], prepared["query"], history, settings.history_turns
        )
        try:
            answer, usage = chat(messages)
        except LLMError as exc:
            raise ServiceError(f"调用模型失败：{exc}") from exc
        return self._finish(prepared, answer, usage, _ms(started))

    def ask_stream(
        self,
        question: str,
        doc_ids: list[str] | None = None,
        top_k: int | None = None,
        strategy: str | None = None,
        rerank: bool | None = None,
        history: list[dict] | None = None,
        mode: str | None = None,
    ):
        """Yield events: pipeline stages, then delta / done / error."""
        prepared: dict = {}
        for event in self.prepare_steps(
            question, doc_ids, top_k, strategy, rerank, history, mode, rewrite=None
        ):
            if isinstance(event, tuple):
                prepared = event[1]
                continue
            yield event

        if not prepared.get("hits"):
            yield {"stage": "delta", "text": NO_CONTEXT_ANSWER}
            yield {"stage": "done", "result": self._finish(prepared, NO_CONTEXT_ANSWER, {}, 0.0)}
            return

        messages = build_chunk_messages(
            prepared["hits"], prepared["query"], history, settings.history_turns
        )
        started = time.perf_counter()
        ttft = None
        usage: dict = {}
        parts: list[str] = []
        try:
            for piece in chat_stream(messages, on_usage=lambda u: usage.update(u)):
                if ttft is None:
                    ttft = _ms(started)
                parts.append(piece)
                yield {"stage": "delta", "text": piece, "ttft_ms": ttft}
        except LLMError as exc:
            yield {"stage": "error", "error": str(exc), "ttft_ms": ttft}
            return

        answer = "".join(parts).strip() or "（模型返回空内容）"
        generate_ms = _ms(started)
        result = self._finish(prepared, answer, usage, generate_ms)
        result["timings"]["ttft_ms"] = ttft or 0.0
        if generate_ms and result["usage"]["completion_tokens"]:
            result["usage"]["tokens_per_second"] = round(
                result["usage"]["completion_tokens"] / (generate_ms / 1000), 1
            )
        yield {"stage": "done", "result": result}

    # ---------- evaluation ----------

    def evaluate(
        self,
        doc_pattern: str = "",
        top_k: int = 5,
        strategy: str = "rrf",
        with_generation: bool = False,
        limit: int | None = None,
        golden_path: str | Path | None = None,
        mode: str | None = None,
        doc_ids: list[str] | None = None,
        questions: list[dict] | None = None,
    ) -> dict:
        from eval.run_eval import load_golden, run_questions

        questions = load_golden(golden_path) if questions is None else questions
        picked = doc_ids if doc_ids is not None else self.doc_ids_matching(doc_pattern)
        rows, skipped = run_questions(
            self,
            questions,
            doc_pattern=doc_pattern,
            top_k=top_k,
            strategy=strategy,
            with_generation=with_generation,
            limit=limit,
            doc_ids=picked,
            mode=mode,
        )
        from eval.metrics import summarize

        return {"rows": rows, "summary": summarize(rows, top_k), "skipped": skipped}

    def generate_questions(
        self,
        doc_ids: list[str] | None = None,
        limit: int = 10,
        on_progress=None,
    ) -> dict:
        from eval.generate import generate_questions

        records = [
            record
            for record in self.library.records
            if doc_ids is None or record.doc_id in doc_ids
        ]
        if not records:
            raise ServiceError("没有可用的块：先上传文档并建索引，再自动出题")
        return generate_questions(records, limit=limit, on_progress=on_progress)

    # ---------- system ----------
    def system_info(self) -> dict:
        return {
            "version": VERSION,
            "started_at": self.started_at,
            "llm": {
                "base_url": settings.llm_base_url,
                "model": settings.llm_model,
                "configured": settings.llm_configured,
                "temperature": settings.temperature,
                "max_tokens": settings.max_tokens,
            },
            "embedding": {
                "model": settings.embed_model,
                "configured": settings.embed_ready,
                "batch": settings.embed_batch,
                "concurrency": settings.embed_concurrency,
            },
            "rerank": {
                "model": settings.rerank_model,
                "enabled": settings.rerank_enabled,
                "configured": settings.rerank_ready,
            },
            "ocr": {
                "enabled": settings.ocr_enabled,
                "ready": settings.ocr_ready,
                # "No key" and "switched off" are different problems and read differently.
                "state": settings.vision_state[1],
                "model": settings.ocr_model,
                "vision_model": settings.vision_model,
                "concurrency": settings.ocr_concurrency,
            },
            "retrieval": {
                "strategy": settings.retrieve_strategy,
                "top_k": settings.retrieve_top_k,
                "candidates": settings.rerank_candidates,
                "rrf_k": settings.rrf_k,
            },
            "chunking": {
                "strategy": settings.chunk_strategy,
                "size": settings.chunk_size,
                "overlap": settings.chunk_overlap,
                "min_chars": settings.chunk_min_chars,
            },
            "index": self.library.stats(),
            "db": self.db.stats(),
            "usage": self.usage_summary(),
            "security": {
                "api_key_required": bool(settings.app_api_key),
                "ui_login": bool(settings.ui_password),
                "rate_limit_per_min": settings.rate_limit_per_min,
            },
        }


def _source_dict(hit: Hit, number: int) -> dict:
    return {
        "n": number,
        "doc_id": hit.record.doc_id,
        "filename": hit.record.filename,
        "page": hit.record.page_from,
        "page_to": hit.record.page_to,
        "kind": hit.record.kind,
        "section": hit.record.section,
        "chars": len(hit.record.text),
        "text": hit.record.text,
        "scores": {
            "fused": round(hit.score, 6),
            "dense": None if hit.dense is None else round(hit.dense, 6),
            "sparse": None if hit.sparse is None else round(hit.sparse, 6),
            "rerank": None if hit.rerank is None else round(hit.rerank, 6),
        },
    }


def _hit_dict(hit: Hit, number: int) -> dict:
    data = _source_dict(hit, number)
    data["preview"] = " ".join(hit.record.text.split())[:120]
    return data
