"""Page-level extraction and quality flags."""

import hashlib
import logging
import re
from datetime import datetime
from pathlib import Path

from rag.config import settings
from rag.ingest.extract import (  # noqa: F401  (re-exported for callers/tests)
    FORMAT_LABELS,
    SUPPORTED_EXT,
    _table_shape_ok,
    extract_pages,
    format_label,
    supported,
)

log = logging.getLogger(__name__)

# Column signal: 2+ spaces or " | ". Density is NOT used -- normal text pages
# already score 3-8 chars per 1000 sq. points, so any low threshold matches all.
_COLUMN_SEP = re.compile(r"\s{2,}|\s\|\s")


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def load_pdf(path: str | Path) -> list[dict]:
    """Extract a PDF page by page (kept for callers that only handle PDF)."""
    try:
        return extract_pages(path)
    except ValueError as exc:
        raise ValueError(f"PDF 解析失败：{exc}") if "解析失败" not in str(exc) else exc


def page_stats(path: str | Path) -> dict[int, dict]:
    """Per-page signals. Kept for callers that want stats without the text."""
    pages = extract_pages(path)
    return {
        page["page"]: {
            "tables": page.get("tables", 0),
            "image_cover": page.get("image_cover", 0.0),
            "vector_ops": page.get("vector_ops", 0),
        }
        for page in pages
    }


def count_tables(path: str | Path) -> dict[int, int]:
    """Count tables per page; failures are per page."""
    return {page: item["tables"] for page, item in page_stats(path).items()}


def column_ratio(text: str) -> float:
    """Share of lines split into 3+ columns."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 5:
        return 0.0
    columnar = sum(1 for line in lines if len(_COLUMN_SEP.split(line)) >= 3)
    return round(columnar / len(lines), 3)


def _signal(stats: dict, page_no: int) -> tuple[int, float, int]:
    """Accept either {page: count} or {page: {tables, image_cover, vector_ops}}."""
    value = stats.get(page_no, 0)
    if isinstance(value, dict):
        return (
            int(value.get("tables", 0)),
            float(value.get("image_cover", 0.0)),
            int(value.get("vector_ops", 0)),
        )
    return int(value or 0), 0.0, 0


def annotate(pages: list[dict], stats: dict) -> list[dict]:
    """Flag pages as scan_like, table_like or figure_like.

    scan_like is about text versus picture area, not about character count: a
    slide with a title and an image of a table is still a scan.
    """
    for page in pages:
        tables, cover, ops = _signal(stats, page["page"])
        page["tables"] = tables
        page["image_cover"] = cover
        page["vector_ops"] = ops
        page.setdefault("column_ratio", column_ratio(page["text"]))

        if page["chars"] < 30:
            page["scan_like"] = True
        elif page["chars"] < settings.scan_page_char_threshold * 2 and cover >= 0.15:
            page["scan_like"] = True
        else:
            page["scan_like"] = False

        page["table_like"] = (
            tables > 0 or page["column_ratio"] >= settings.table_column_ratio
        )
        page["figure_like"] = (
            not page["scan_like"]
            and not page["table_like"]
            and (
                cover >= 0.05
                or (ops >= 40 and page["chars"] < settings.figure_text_max)
            )
        )
    return pages


def load_document(
    path: str | Path,
    doc_id: str | None = None,
    filename: str | None = None,
) -> dict:
    """Parse any supported file into the document object shared by UI and API."""
    path = Path(path)
    if not path.exists():
        raise ValueError(f"文件不存在：{path}")

    doc_id = doc_id or sha256_file(path)[:16]
    pages = extract_pages(path)
    annotate(pages, {page["page"]: page for page in pages})

    return {
        "id": doc_id,
        "filename": filename or path.name,
        "path": str(path),
        "format": format_label(path),
        "pages": pages,
        "page_count": len(pages),
        "chars": sum(p["chars"] for p in pages),
        "scan_like_pages": [p["page"] for p in pages if p["scan_like"]],
        "table_like_pages": [p["page"] for p in pages if p["table_like"]],
        "figure_like_pages": [p["page"] for p in pages if p["figure_like"]],
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }


def page_text(doc: dict, page_no: int) -> str:
    """Render one page for preview."""
    for page in doc["pages"]:
        if page["page"] == page_no:
            header = (
                f"第 {page_no} 页 · {page['chars']} 字符 · 表格 {page['tables']} 个"
                f" · 列结构 {page.get('column_ratio', 0):.2f}"
                f" · 图像覆盖 {page.get('image_cover', 0):.0%}"
            )
            body = page["text"] or "（本页无可提取文本，疑似扫描页）"
            return f"{header}\n{'-' * 48}\n{body}"
    return f"第 {page_no} 页不存在（文档共 {doc['page_count']} 页）"
