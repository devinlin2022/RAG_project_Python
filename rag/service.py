"""Single orchestration layer: ingest, index, retrieve, answer, evaluate.

Both the Gradio UI and the HTTP API call this module. Nothing here knows about
either transport, which is what keeps one process and one index state.
"""

import logging
import time
from datetime import datetime
from pathlib import Path

from rag.chunk.registry import chunk_document
from rag.config import VERSION, settings
from rag.embed.client import EmbedError, embed_texts
from rag.generate.llm import LLMError, chat, chat_stream
from rag.generate.prompt import NO_CONTEXT_ANSWER, build_chunk_messages
from rag.generate.rewrite import rewrite_question
from rag.index.library import ChunkLibrary
from rag.ingest.enhance import Block, all_blocks, enhance
from rag.ingest.loader import load_document, sha256_file
from rag.retrieve import reranker
from rag.retrieve.retriever import Hit, diversify, rerank_hits, search
from rag.store.db import Database

log = logging.getLogger(__name__)

STRATEGY_LABELS = {
    "rrf": "RRF 融合",
    "dense": "向量",
    "sparse": "BM25",
}


class ServiceError(ValueError):
    """Invalid request; the message is safe to show."""


def _ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 1)


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


class Service:
    def __init__(self, db: Database | None = None) -> None:
        self.db = db or Database(settings.db_path)
        self.library = ChunkLibrary(self.db)
        self.docs: dict[str, dict] = {}
        self.started_at = _now()
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
                "pages": pages,
                "page_count": len(pages),
                "chars": sum(p["chars"] for p in pages),
                "scan_like_pages": [p["page"] for p in pages if p["scan_like"]],
                "table_like_pages": [p["page"] for p in pages if p["table_like"]],
                "figure_like_pages": [p["page"] for p in pages if p["figure_like"]],
                "created_at": meta["created_at"],
                "strategy": meta["strategy"],
                "chunk_count": meta["chunk_count"],
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
        """Parse and persist a PDF. Returns (doc, was_already_present)."""
        path = Path(path)
        doc_id = sha256_file(path)[:16]
        if doc_id in self.docs:
            return self.docs[doc_id], True

        doc = load_document(path, doc_id=doc_id, filename=filename)
        doc["dim"] = 0
        self.db.upsert_doc(self._meta(doc))
        self.db.save_pages(doc_id, doc["pages"])
        self.docs[doc_id] = doc
        log.info("ingest doc=%s pages=%d chars=%d", doc_id, doc["page_count"], doc["chars"])
        return doc, False

    def get(self, doc_id: str | None) -> dict:
        if not doc_id or doc_id not in self.docs:
            raise ServiceError("文档不存在，请先上传并解析 PDF")
        return self.docs[doc_id]

    def list_documents(self) -> list[dict]:
        rows = []
        for doc in self.docs.values():
            rows.append(
                {
                    "id": doc["id"],
                    "filename": doc["filename"],
                    "pages": doc["page_count"],
                    "chars": doc["chars"],
                    "chunks": doc.get("chunk_count", 0),
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
        if doc.get("chunk_count") and doc.get("indexed_at") and not force:
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

    # ---------- retrieval ----------

    def _embed_query(self, query: str) -> tuple[list[float] | None, float, str]:
        if not settings.embed_ready or not len(self.library.dense):
            return None, 0.0, ""
        started = time.perf_counter()
        try:
            return embed_texts([query])[0], _ms(started), ""
        except EmbedError as exc:
            return None, _ms(started), str(exc)

    def retrieve(
        self,
        query: str,
        doc_ids: list[str] | None = None,
        top_k: int | None = None,
        strategy: str | None = None,
        rerank: bool | None = None,
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
            candidates=settings.rerank_candidates,
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

    def _prepare(
        self,
        question: str,
        doc_ids: list[str] | None,
        top_k: int | None,
        strategy: str | None,
        rerank: bool | None,
        history: list[dict] | None,
    ) -> dict:
        started = time.perf_counter()
        rewritten, was_rewritten, rewrite_error = rewrite_question(
            question,
            history,
            chat,
            turns=settings.history_turns,
            enabled=settings.rewrite_enabled and settings.llm_configured,
        )
        rewrite_ms = _ms(started)

        result = self.retrieve(rewritten, doc_ids, top_k=top_k, strategy=strategy, rerank=rerank)
        # retrieve_ms already covers embedding + both rails + fusion + rerank, so
        # the stage values below are components, not siblings to be summed.
        timings = {"rewrite_ms": rewrite_ms, **result["timings"]}
        return {
            "question": question.strip(),
            "query": rewritten,
            "rewritten": was_rewritten,
            "rewrite_error": rewrite_error,
            "hits": result["hits"],
            "timings": timings,
            "rerank": result["rerank"],
            "degraded": result["degraded"],
            "embed_error": result["embed_error"],
            "candidates": result["candidates"],
            "strategy": result["strategy"],
        }

    def _finish(self, prepared: dict, answer: str, usage: dict, generate_ms: float) -> dict:
        hits = prepared["hits"]
        from eval.metrics import citation_grounding, is_abstain

        valid, total = citation_grounding(answer, len(hits))
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
            "rerank": prepared["rerank"],
            "degraded": prepared["degraded"],
            "embed_error": prepared["embed_error"],
            "abstained": is_abstain(answer),
            "citations": {"valid": valid, "total": total},
            "usage": usage,
            "model": settings.llm_model,
        }
        self.db.log_query(
            question=prepared["question"],
            answer=answer,
            pages=sorted({hit.record.page_from for hit in hits}),
            ms=int(payload["timings"]["total_ms"]),
            top_k=len(hits),
            strategy=prepared["strategy"],
            rewritten=prepared["query"] if prepared["rewritten"] else "",
            reranked=bool(prepared["rerank"]["used"]),
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
    ) -> dict:
        prepared = self._prepare(question, doc_ids, top_k, strategy, rerank, history)
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
    ):
        """Yield events: prepared / delta / done / error."""
        prepared = self._prepare(question, doc_ids, top_k, strategy, rerank, history)
        yield {
            "stage": "prepared",
            "query": prepared["query"],
            "rewritten": prepared["rewritten"],
            "sources": [_source_dict(hit, number) for number, hit in enumerate(prepared["hits"], 1)],
            "timings": prepared["timings"],
            "rerank": prepared["rerank"],
            "degraded": prepared["degraded"],
        }

        if not prepared["hits"]:
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
        result = self._finish(prepared, answer, usage, _ms(started))
        result["timings"]["ttft_ms"] = ttft or 0.0
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
    ) -> dict:
        from eval.run_eval import load_golden, run_questions

        questions = load_golden(golden_path)
        doc_ids = self.doc_ids_matching(doc_pattern)
        rows, skipped = run_questions(
            self,
            questions,
            doc_pattern=doc_pattern,
            top_k=top_k,
            strategy=strategy,
            with_generation=with_generation,
            limit=limit,
            doc_ids=doc_ids,
        )
        from eval.metrics import summarize

        return {"rows": rows, "summary": summarize(rows, top_k), "skipped": skipped}

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
            },
            "rerank": {
                "model": settings.rerank_model,
                "enabled": settings.rerank_enabled,
                "configured": settings.rerank_ready,
            },
            "ocr": {
                "enabled": settings.ocr_enabled,
                "model": settings.ocr_model,
                "vision_model": settings.vision_model,
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
