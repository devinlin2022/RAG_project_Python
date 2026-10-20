"""Page-level PDF extraction with quality flags.

Uses extract_pages instead of extract_text: ~25% slower, but keeps page numbers,
which later versions need for citations, deletion and evaluation.
"""

import hashlib
import logging
import re
from datetime import datetime
from pathlib import Path

from rag.config import settings

log = logging.getLogger(__name__)

# Zero-width chars inflate char counts and break scan-page detection.
_INVISIBLE = re.compile(r"[\u200b-\u200f\ufeff]")

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
    """Extract text page by page."""
    from pdfminer.high_level import extract_pages
    from pdfminer.layout import LAParams, LTTextContainer

    pages: list[dict] = []
    page_no = 0
    laparams = LAParams(detect_vertical=True, char_margin=2.0, line_margin=0.3)

    try:
        for layout in extract_pages(str(path), laparams=laparams):
            page_no += 1
            text = "".join(
                element.get_text()
                for element in layout
                if isinstance(element, LTTextContainer)
            )
            text = _INVISIBLE.sub("", text)
            area = max(float(layout.width) * float(layout.height), 1.0)
            cleaned = text.strip()
            pages.append(
                {
                    "page": page_no,
                    "text": cleaned,
                    "chars": len(cleaned),
                    "density": round(len(cleaned) / area * 1000, 3),
                }
            )
    except Exception as exc:  # encrypted, corrupted or not a PDF
        raise ValueError(f"PDF 解析失败：{exc}") from exc

    if not pages:
        raise ValueError("未能从该 PDF 中提取任何页面，可能文件为空或已损坏")
    return pages


def count_tables(path: str | Path) -> dict[int, int]:
    """Count tables per page via pdfplumber; failures are per page."""
    try:
        import pdfplumber
    except ImportError:
        log.warning("pdfplumber not installed, table detection skipped")
        return {}

    counts: dict[int, int] = {}
    try:
        with pdfplumber.open(str(path)) as pdf:
            for index, page in enumerate(pdf.pages, start=1):
                try:
                    counts[index] = len(page.find_tables())
                except Exception:
                    counts[index] = 0
    except Exception as exc:
        log.warning("table detection failed: %s", exc)
        return {}
    return counts


def column_ratio(text: str) -> float:
    """Share of lines split into 3+ columns."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 5:
        return 0.0
    columnar = sum(1 for line in lines if len(_COLUMN_SEP.split(line)) >= 3)
    return round(columnar / len(lines), 3)


def annotate(pages: list[dict], tables: dict[int, int]) -> list[dict]:
    """Flag pages as scan_like / table_like."""
    for page in pages:
        page["tables"] = tables.get(page["page"], 0)
        page["column_ratio"] = column_ratio(page["text"])
        page["scan_like"] = page["chars"] < settings.scan_page_char_threshold
        page["table_like"] = (
            page["tables"] > 0
            or page["column_ratio"] >= settings.table_column_ratio
        )
    return pages


def load_document(
    path: str | Path,
    doc_id: str | None = None,
    filename: str | None = None,
) -> dict:
    """Parse a PDF into the document object shared by UI and API."""
    path = Path(path)
    if not path.exists():
        raise ValueError(f"文件不存在：{path}")

    doc_id = doc_id or sha256_file(path)[:16]
    pages = annotate(load_pdf(path), count_tables(path))

    return {
        "id": doc_id,
        "filename": filename or path.name,
        "path": str(path),
        "pages": pages,
        "page_count": len(pages),
        "chars": sum(p["chars"] for p in pages),
        "scan_like_pages": [p["page"] for p in pages if p["scan_like"]],
        "table_like_pages": [p["page"] for p in pages if p["table_like"]],
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }


def page_text(doc: dict, page_no: int) -> str:
    """Render one page for preview."""
    for page in doc["pages"]:
        if page["page"] == page_no:
            header = (
                f"第 {page_no} 页 · {page['chars']} 字符 · 表格 {page['tables']} 个"
                f" · 列结构 {page.get('column_ratio', 0):.2f}"
            )
            return f"{header}\n{'-' * 48}\n{page['text'] or '（本页无可提取文本，疑似扫描页）'}"
    return f"第 {page_no} 页不存在（文档共 {doc['page_count']} 页）"
