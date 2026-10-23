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


def _table_shape_ok(table, page_area: float) -> bool:
    """Reject layout boxes that pdfplumber reports as tables.

    A slide's coloured content blocks look exactly like a 2x2 table to the line
    detector. Three cheap tests separate them from real tables:

    - a table needs at least 3 rows or 3 columns (a 2x2 box is a layout)
    - a single column is never a table
    - a table covering ~the whole page is the page background, not a grid
    """
    try:
        rows = len(table.rows)
        cols = max((len(row.cells) for row in table.rows), default=0)
    except Exception:
        return False
    if rows < 2 or cols < 2:
        return False
    if rows < 3 and cols < 3:
        return False
    try:
        x0, top, x1, bottom = table.bbox
        area = max((x1 - x0) * (bottom - top), 0.0)
        if page_area and area / page_area >= 0.85:
            return False
    except Exception:
        pass
    return True


def page_stats(path: str | Path) -> dict[int, dict]:
    """One pdfplumber pass: table count, image coverage, vector-op count per page.

    Kept in a single pass because pdfplumber parsing dominates ingest time on long
    documents, and every downstream decision needs the same three numbers.
    """
    stats: dict[int, dict] = {}
    try:
        import pdfplumber
    except ImportError:
        log.warning("pdfplumber not installed, table/figure detection skipped")
        return stats

    try:
        with pdfplumber.open(str(path)) as pdf:
            for index, page in enumerate(pdf.pages, start=1):
                try:
                    area = max(float(page.width) * float(page.height), 1.0)
                    tables = sum(
                        1 for table in page.find_tables() if _table_shape_ok(table, area)
                    )
                    cover = min(
                        sum(
                            float(image["width"]) * float(image["height"])
                            for image in (page.images or [])
                        )
                        / area,
                        1.0,
                    )
                    ops = (
                        len(page.rects or [])
                        + len(page.lines or [])
                        + len(page.curves or [])
                    )
                    stats[index] = {
                        "tables": tables,
                        "image_cover": round(cover, 3),
                        "vector_ops": ops,
                    }
                except Exception:
                    stats[index] = {"tables": 0, "image_cover": 0.0, "vector_ops": 0}
    except Exception as exc:
        log.warning("page stats failed: %s", exc)
    return stats


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
    """Flag pages as scan_like / table_like / figure_like.

    scan_like no longer means "little text". A slide with 170 chars of bullets is
    not a scan, and a slide whose table is an embedded image is one even though it
    has 170 chars of title. The signal is text *versus picture area*.
    """
    for page in pages:
        tables, cover, ops = _signal(stats, page["page"])
        page["tables"] = tables
        page["image_cover"] = cover
        page["vector_ops"] = ops
        page["column_ratio"] = column_ratio(page["text"])

        if page["chars"] < 30:
            page["scan_like"] = True
        elif (
            page["chars"] < settings.scan_page_char_threshold * 2 and cover >= 0.15
        ):
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
    """Parse a PDF into the document object shared by UI and API."""
    path = Path(path)
    if not path.exists():
        raise ValueError(f"文件不存在：{path}")

    doc_id = doc_id or sha256_file(path)[:16]
    pages = annotate(load_pdf(path), page_stats(path))

    return {
        "id": doc_id,
        "filename": filename or path.name,
        "path": str(path),
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
            )
            return f"{header}\n{'-' * 48}\n{page['text'] or '（本页无可提取文本，疑似扫描页）'}"
    return f"第 {page_no} 页不存在（文档共 {doc['page_count']} 页）"
