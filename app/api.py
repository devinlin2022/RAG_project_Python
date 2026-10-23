"""HTTP layer: request parsing, status codes, error mapping only."""

import logging
import time
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.state import STORE
from rag.config import VERSION, settings
from rag.chunk.registry import chunk_document, compare_strategies
from rag.chunk.profile import profile_document
from rag.generate.answer import PageAnswerError, answer_by_pages
from rag.generate.llm import LLMError
from rag.ingest.enhance import all_blocks, enhance
from rag.ingest.loader import load_document, sha256_bytes
from rag.utils import format_pages

log = logging.getLogger(__name__)
router = APIRouter()


class IngestResponse(BaseModel):
    doc_id: str
    filename: str
    pages: int
    chars: int
    scan_like_pages: list[int]
    table_like_pages: list[int]
    elapsed_ms: int


class PageMetric(BaseModel):
    page: int
    chars: int
    density: float
    tables: int
    column_ratio: float
    scan_like: bool
    table_like: bool


class AskRequest(BaseModel):
    doc_id: str
    pages: list[int] = Field(default_factory=list)
    question: str


class AskResponse(BaseModel):
    answer: str
    pages: list[int]
    chars: int
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


@router.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "version": VERSION,
        "docs": len(STORE),
        "llm_configured": settings.llm_configured,
        "llm_base_url": settings.llm_base_url,
        "llm_model": settings.llm_model,
        "ocr_enabled": settings.ocr_enabled,
        "ocr_ready": settings.ocr_ready,
        "ocr_model": settings.ocr_model,
        "vision_model": settings.vision_model,
        "embed_ready": settings.embed_ready,
        "embed_model": settings.embed_model,
    }


@router.post("/ingest", response_model=IngestResponse)
async def ingest(file: UploadFile = File(...)) -> IngestResponse:
    filename = file.filename or "document.pdf"
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="本版仅支持 PDF")

    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=400, detail="上传内容为空")

    started = time.perf_counter()
    doc_id = sha256_bytes(payload)[:16]
    target = settings.upload_dir / f"{doc_id}_{Path(filename).name}"
    target.write_bytes(payload)

    try:
        doc = load_document(target, doc_id=doc_id, filename=filename)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    STORE.put(doc)
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    log.info(
        "ingest doc=%s pages=%d chars=%d ms=%d",
        doc["id"], doc["page_count"], doc["chars"], elapsed_ms,
    )
    return IngestResponse(
        doc_id=doc["id"],
        filename=doc["filename"],
        pages=doc["page_count"],
        chars=doc["chars"],
        scan_like_pages=doc["scan_like_pages"],
        table_like_pages=doc["table_like_pages"],
        elapsed_ms=elapsed_ms,
    )


@router.get("/docs/{doc_id}/pages", response_model=list[PageMetric])
def list_pages(doc_id: str) -> list[PageMetric]:
    doc = STORE.get(doc_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="文档不存在，请先调用 /ingest")
    return [
        PageMetric(
            page=p["page"],
            chars=p["chars"],
            density=p["density"],
            tables=p["tables"],
            column_ratio=p.get("column_ratio", 0.0),
            scan_like=p["scan_like"],
            table_like=p["table_like"],
        )
        for p in doc["pages"]
    ]


@router.post("/ask-by-pages", response_model=AskResponse)
def ask_by_pages(payload: AskRequest) -> AskResponse:
    doc = STORE.get(payload.doc_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="文档不存在，请先调用 /ingest")

    try:
        result = answer_by_pages(doc, payload.pages, payload.question)
    except PageAnswerError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except LLMError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    log.info("ask doc=%s pages=%s", doc["id"], format_pages(result["pages"]))
    return AskResponse(**result)


class EnhanceRequest(BaseModel):
    limit_pages: int | None = None


class ChunkRequest(BaseModel):
    doc_id: str
    strategy: str = "auto"
    size: int = 500
    overlap: int = 80
    min_chars: int = 120
    percentile: float = 85.0


def _require(doc_id: str) -> dict:
    doc = STORE.get(doc_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="文档不存在，请先调用 /ingest")
    return doc


def _pairs(doc: dict) -> list[tuple[int, object]]:
    return all_blocks(doc)


@router.post("/docs/{doc_id}/enhance")
def run_enhance(doc_id: str, payload: EnhanceRequest | None = None) -> dict:
    """Run the OCR / table / figure rails and return the per-page report."""
    doc = _require(doc_id)
    request = payload or EnhanceRequest()
    report = enhance(doc, limit_pages=request.limit_pages)
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


@router.get("/docs/{doc_id}/profile")
def get_profile(doc_id: str) -> dict:
    """Document census plus the recommended chunking strategy."""
    doc = _require(doc_id)
    profile = profile_document(
        _pairs(doc),
        pages=doc["page_count"],
        table_pages=len(doc["table_like_pages"]),
    )
    return profile.to_dict()


@router.post("/chunk")
def run_chunk(payload: ChunkRequest) -> dict:
    """Chunk a parsed document; `auto` profiles first, then dispatches."""
    doc = _require(payload.doc_id)
    result = chunk_document(
        _pairs(doc),
        strategy=payload.strategy,
        size=payload.size,
        overlap=payload.overlap,
        min_chars=payload.min_chars,
        percentile=payload.percentile,
        pages=doc["page_count"],
        table_pages=len(doc["table_like_pages"]),
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
    """Run all four strategies and return comparable structural metrics."""
    doc = _require(payload.doc_id)
    rows = compare_strategies(
        _pairs(doc),
        size=payload.size,
        overlap=payload.overlap,
        min_chars=payload.min_chars,
        percentile=payload.percentile,
        pages=doc["page_count"],
        table_pages=len(doc["table_like_pages"]),
    )
    return {"doc_id": doc["id"], "rows": rows}
