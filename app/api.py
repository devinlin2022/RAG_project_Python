"""HTTP layer: request parsing, status codes, error mapping only."""

import json
import logging
import time
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.auth import rate_limit, require_key
from app.state import get_service
from rag.chunk.registry import chunk_document, compare_strategies
from rag.chunk.profile import profile_document
from rag.config import VERSION, settings
from rag.generate.answer import PageAnswerError, answer_by_pages
from rag.generate.llm import LLMError
from rag.ingest.enhance import all_blocks
from rag.ingest.loader import sha256_bytes
from rag.service import ServiceError
from rag.utils import format_pages

log = logging.getLogger(__name__)

public = APIRouter()
router = APIRouter(dependencies=[Depends(require_key)])


# ---------- schemas ----------


class IngestResponse(BaseModel):
    doc_id: str
    filename: str
    pages: int
    chars: int
    scan_like_pages: list[int]
    table_like_pages: list[int]
    deduplicated: bool = False


class AskRequest(BaseModel):
    question: str
    doc_ids: list[str] | None = None
    top_k: int = Field(default=6, ge=1, le=30)
    strategy: str = "rrf"
    rerank: bool | None = None
    history: list[dict] = Field(default_factory=list)


class RetrieveRequest(BaseModel):
    query: str
    doc_ids: list[str] | None = None
    top_k: int = Field(default=5, ge=1, le=20)


class IndexRequest(BaseModel):
    strategy: str = "auto"
    size: int | None = None
    overlap: int | None = None
    force: bool = False


class EnhanceRequest(BaseModel):
    limit_pages: int | None = None


class ChunkRequest(BaseModel):
    doc_id: str
    strategy: str = "auto"
    size: int = 500
    overlap: int = 80
    min_chars: int = 120
    percentile: float = 85.0


class EvalRequest(BaseModel):
    doc_pattern: str = ""
    top_k: int = Field(default=5, ge=1, le=20)
    strategy: str = "rrf"
    with_generation: bool = False
    limit: int | None = None


# ---------- health / system ----------


@public.get("/health")
def health() -> dict:
    service = get_service()
    return {
        "status": "ok",
        "version": VERSION,
        "docs": len(service.docs),
        "chunks": service.library.stats()["chunks"],
        "llm_configured": settings.llm_configured,
        "llm_model": settings.llm_model,
        "embed_ready": settings.embed_ready,
        "rerank_ready": settings.rerank_ready,
        "ocr_ready": settings.ocr_ready,
        "auth_required": bool(settings.app_api_key),
    }


@router.get("/system")
def system_info() -> dict:
    return get_service().system_info()


@router.get("/queries")
def recent_queries(limit: int = Query(default=20, ge=1, le=100)) -> list[dict]:
    return get_service().db.recent_queries(limit)


# ---------- documents ----------


def _require_doc(doc_id: str) -> dict:
    try:
        return get_service().get(doc_id)
    except ServiceError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/ingest", response_model=IngestResponse, dependencies=[Depends(rate_limit)])
async def ingest(file: UploadFile = File(...)) -> IngestResponse:
    filename = file.filename or "document.pdf"
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="本版仅支持 PDF")

    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=400, detail="上传内容为空")
    if len(payload) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"文件超过 {settings.max_upload_mb} MB 上限")

    doc_id = sha256_bytes(payload)[:16]
    target = settings.upload_dir / f"{doc_id}_{Path(filename).name}"
    target.write_bytes(payload)

    try:
        doc, existed = get_service().ingest(target, filename)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return IngestResponse(
        doc_id=doc["id"],
        filename=doc["filename"],
        pages=doc["page_count"],
        chars=doc["chars"],
        scan_like_pages=doc["scan_like_pages"],
        table_like_pages=doc["table_like_pages"],
        deduplicated=existed,
    )


@router.get("/documents")
def list_docs() -> list[dict]:
    return get_service().list_documents()


@router.delete("/documents/{doc_id}")
def delete_doc(doc_id: str) -> dict:
    if not get_service().delete(doc_id):
        raise HTTPException(status_code=404, detail="文档不存在")
    return {"deleted": doc_id}


@router.get("/documents/{doc_id}/pages")
def list_pages(doc_id: str) -> list[dict]:
    doc = _require_doc(doc_id)
    return [
        {
            "page": page["page"],
            "chars": page["chars"],
            "density": page["density"],
            "tables": page["tables"],
            "column_ratio": page.get("column_ratio", 0.0),
            "scan_like": page["scan_like"],
            "table_like": page["table_like"],
            "figure_like": page["figure_like"],
        }
        for page in doc["pages"]
    ]


@router.post("/documents/{doc_id}/index")
def index_doc(doc_id: str, payload: IndexRequest | None = None) -> dict:
    _require_doc(doc_id)
    request = payload or IndexRequest()
    try:
        return get_service().index(
            doc_id,
            strategy=request.strategy,
            size=request.size,
            overlap=request.overlap,
            force=request.force,
        )
    except ServiceError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/documents/{doc_id}/enhance")
def run_enhance(doc_id: str, payload: EnhanceRequest | None = None) -> dict:
    doc = _require_doc(doc_id)
    request = payload or EnhanceRequest()
    report = get_service().enhance(doc_id, limit_pages=request.limit_pages)
    return {
        "doc_id": doc["id"],
        "ms": report["ms"],
        "api_calls": report["api_calls"],
        "skipped": report["skipped"],
        "text_before": report["text_before"],
        "text_after": report["text_after"],
        "pages": [
            {
                "page": result.page,
                "routes": result.routes,
                "status": result.status,
                "chars_before": result.chars_before,
                "chars_after": result.chars_after,
                "ms": result.ms,
                "engine": result.engine,
                "error": result.error,
            }
            for result in report["pages"]
        ],
    }


@router.get("/documents/{doc_id}/profile")
def get_profile(doc_id: str) -> dict:
    doc = _require_doc(doc_id)
    profile = profile_document(
        all_blocks(doc),
        pages=doc["page_count"],
        table_pages=len(doc["table_like_pages"]),
        figure_pages=len(doc["figure_like_pages"]),
    )
    return profile.to_dict()


@router.post("/chunk")
def run_chunk(payload: ChunkRequest) -> dict:
    doc = _require_doc(payload.doc_id)
    result = chunk_document(
        all_blocks(doc),
        strategy=payload.strategy,
        size=payload.size,
        overlap=payload.overlap,
        min_chars=payload.min_chars,
        percentile=payload.percentile,
        pages=doc["page_count"],
        table_pages=len(doc["table_like_pages"]),
        figure_pages=len(doc["figure_like_pages"]),
    )
    return {
        "doc_id": doc["id"],
        "strategy": result["strategy"],
        "size": result.get("size"),
        "overlap": result.get("overlap"),
        "ms": result["ms"],
        "info": result.get("info", {}),
        "metrics": result.get("metrics", {}),
        "chunks": [chunk.to_dict() for chunk in result["chunks"]],
    }


@router.post("/chunk/compare")
def run_compare(payload: ChunkRequest) -> dict:
    doc = _require_doc(payload.doc_id)
    rows = compare_strategies(
        all_blocks(doc),
        size=payload.size,
        overlap=payload.overlap,
        min_chars=payload.min_chars,
        percentile=payload.percentile,
        pages=doc["page_count"],
        table_pages=len(doc["table_like_pages"]),
        figure_pages=len(doc["figure_like_pages"]),
    )
    return {"doc_id": doc["id"], "rows": rows}


# ---------- retrieval / answering ----------


@router.post("/retrieve", dependencies=[Depends(rate_limit)])
def retrieve(payload: RetrieveRequest) -> dict:
    try:
        return get_service().debug(payload.query, payload.doc_ids, payload.top_k)
    except ServiceError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/ask", dependencies=[Depends(rate_limit)])
def ask(payload: AskRequest) -> dict:
    try:
        return get_service().ask(
            payload.question,
            doc_ids=payload.doc_ids,
            top_k=payload.top_k,
            strategy=payload.strategy,
            rerank=payload.rerank,
            history=payload.history,
        )
    except ServiceError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/ask/stream", dependencies=[Depends(rate_limit)])
def ask_stream(request: Request, payload: AskRequest) -> StreamingResponse:
    """Server-sent events: one JSON object per event, stages prepared/delta/done/error."""

    def events():
        try:
            for event in get_service().ask_stream(
                payload.question,
                doc_ids=payload.doc_ids,
                top_k=payload.top_k,
                strategy=payload.strategy,
                rerank=payload.rerank,
                history=payload.history,
            ):
                if event["stage"] == "prepared":
                    event = {**event, "stage": "prepared"}
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except ServiceError as exc:
            yield f"data: {json.dumps({'stage': 'error', 'error': str(exc)}, ensure_ascii=False)}\n\n"
        except Exception as exc:  # never leave the client hanging
            log.exception("stream failed")
            yield f"data: {json.dumps({'stage': 'error', 'error': str(exc)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


class PageAskRequest(BaseModel):
    doc_id: str
    pages: list[int] = Field(default_factory=list)
    question: str


@router.post("/ask-by-pages", dependencies=[Depends(rate_limit)])
def ask_by_pages(payload: PageAskRequest) -> dict:
    """V1 endpoint kept for compatibility: answer from hand-picked pages."""
    doc = _require_doc(payload.doc_id)
    try:
        result = answer_by_pages(doc, payload.pages, payload.question)
    except PageAnswerError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except LLMError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    log.info("ask-by-pages doc=%s pages=%s", doc["id"], format_pages(result["pages"]))
    return result


# ---------- evaluation ----------


@router.post("/eval/run", dependencies=[Depends(rate_limit)])
def run_eval(payload: EvalRequest) -> dict:
    return get_service().evaluate(
        doc_pattern=payload.doc_pattern,
        top_k=payload.top_k,
        strategy=payload.strategy,
        with_generation=payload.with_generation,
        limit=payload.limit,
    )


# ---------- misc ----------


@router.get("/metrics/summary")
def metrics_summary() -> dict:
    service = get_service()
    started = time.perf_counter()
    return {
        "index": service.library.stats(),
        "db": service.db.stats(),
        "ms": round((time.perf_counter() - started) * 1000, 1),
    }
