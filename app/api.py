"""HTTP layer: request parsing, status codes, error mapping only."""

import logging
import time
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.state import STORE
from rag.config import VERSION, settings
from rag.generate.answer import PageAnswerError, answer_by_pages
from rag.generate.llm import LLMError
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
