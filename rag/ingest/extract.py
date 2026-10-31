"""Multi-format extraction, fast path first.

One contract for every file type: a list of page dicts with `page`, `text`,
`chars`, `density` and the signals the router needs (`tables`, `image_cover`,
`vector_ops`). PDF keeps real page numbers; formats without pages are cut into
virtual pages so citations, deletion and evaluation behave the same.
"""

import csv
import io
import logging
import re
from pathlib import Path

from rag.config import settings

log = logging.getLogger(__name__)

SUPPORTED_EXT = (
    ".pdf",
    ".docx",
    ".pptx",
    ".xlsx",
    ".csv",
    ".md",
    ".markdown",
    ".txt",
    ".html",
    ".htm",
)

FORMAT_LABELS = {
    ".pdf": "PDF",
    ".docx": "Word",
    ".pptx": "PPT",
    ".xlsx": "Excel",
    ".csv": "CSV",
    ".md": "Markdown",
    ".markdown": "Markdown",
    ".txt": "文本",
    ".html": "HTML",
    ".htm": "HTML",
}

# Zero-width chars inflate char counts and break scan-page detection.
_INVISIBLE = re.compile(r"[\u200b-\u200f\ufeff]")
_MULTI_NEWLINE = re.compile(r"\n{3,}")
_HTML_TAG = re.compile(r"<[^>]+>")

# A row of cells is worth keeping as a table only if it has this many columns.
_TABLE_MIN_COLS = 2


def supported(path: str | Path) -> bool:
    return Path(path).suffix.lower() in SUPPORTED_EXT


def format_label(path: str | Path) -> str:
    return FORMAT_LABELS.get(Path(path).suffix.lower(), Path(path).suffix.lstrip(".").upper() or "文件")


def clean_text(text: str) -> str:
    """Normalise extraction artefacts without touching the wording."""
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    text = _INVISIBLE.sub("", text)
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    return _MULTI_NEWLINE.sub("\n\n", text).strip()


def _page_dict(page: int, text: str, area: float = 0.0) -> dict:
    cleaned = clean_text(text)
    density = round(len(cleaned) / area * 1000, 3) if area else round(len(cleaned) / 1000, 3)
    return {
        "page": page,
        "text": cleaned,
        "chars": len(cleaned),
        "density": density,
        "tables": 0,
        "image_cover": 0.0,
        "vector_ops": 0,
    }


# ---------------------------------------------------------------- pdf


def _object_area(obj) -> float:
    try:
        left, bottom, right, top = obj.get_pos()
        return max((right - left) * (top - bottom), 0.0)
    except Exception:
        return 0.0


def _pdf_with_pdfium(path: Path) -> list[dict]:
    import pypdfium2 as pdfium
    from pypdfium2.raw import FPDF_PAGEOBJ_IMAGE, FPDF_PAGEOBJ_PATH

    pages: list[dict] = []
    document = pdfium.PdfDocument(str(path))
    try:
        total = len(document)
        for index in range(total):
            page = document[index]
            try:
                textpage = page.get_textpage()
                text = textpage.get_text_bounded()
            except Exception:  # very old pdfium builds
                text = page.get_textpage().get_text_range()

            width, height = page.get_size()
            area = max(width * height, 1.0)
            try:
                images = list(page.get_objects(filter=(FPDF_PAGEOBJ_IMAGE,)))
                cover = min(sum(_object_area(obj) for obj in images) / area, 1.0)
            except Exception:
                cover = 0.0
            try:
                ops = len(list(page.get_objects(filter=(FPDF_PAGEOBJ_PATH,))))
            except Exception:
                ops = 0

            item = _page_dict(index + 1, text, area)
            item["image_cover"] = round(cover, 3)
            item["vector_ops"] = ops
            pages.append(item)
    finally:
        document.close()
    return pages


def _pdf_with_pdfminer(path: Path) -> list[dict]:
    from pdfminer.high_level import extract_pages
    from pdfminer.layout import LAParams, LTTextContainer

    pages: list[dict] = []
    laparams = LAParams(detect_vertical=True, char_margin=2.0, line_margin=0.3)
    for index, layout in enumerate(extract_pages(str(path), laparams=laparams), start=1):
        text = "".join(
            element.get_text() for element in layout if isinstance(element, LTTextContainer)
        )
        area = max(float(layout.width) * float(layout.height), 1.0)
        pages.append(_page_dict(index, text, area))
    return pages


def _column_ratio(text: str) -> float:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 5:
        return 0.0
    columnar = sum(1 for line in lines if len(re.split(r"\s{2,}|\s\|\s", line)) >= 3)
    return round(columnar / len(lines), 3)


def looks_tabular(page: dict) -> bool:
    """Cheap pre-check before the ruled-table scan.

    Text layout alone is not enough: pdfium renders a ruled table with single
    spaces between cells. The path count is the giveaway.
    """
    text = page["text"]
    if not text.strip():
        return False
    if page.get("vector_ops", 0) >= 8:
        return True
    if _column_ratio(text) >= 0.25:
        return True
    return text.count("|") >= 4


def _table_shape_ok(table, page_area: float) -> bool:
    """Reject layout boxes that pdfplumber reports as tables.

    A table needs at least 3 rows or 3 columns, more than one column, and must not
    cover the whole page.
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


def scan_ruled_tables(path: Path, pages: list[dict]) -> int:
    """Count recognised tables, only on pages that look tabular.

    Candidates come from the text layout, so a prose document costs no extra pass.
    """
    try:
        import pdfplumber
    except ImportError:
        return 0

    candidates = [p for p in pages if looks_tabular(p)]
    budget = max(settings.table_scan_max_pages, 0)
    if budget and len(candidates) > budget:
        # Prefer pages with a stronger column signal when over budget.
        candidates = sorted(candidates, key=lambda p: -_column_ratio(p["text"]))[:budget]

    if not candidates:
        return 0

    scanned = 0
    try:
        with pdfplumber.open(str(path)) as pdf:
            for page in candidates:
                if not 1 <= page["page"] <= len(pdf.pages):
                    continue
                try:
                    raw = pdf.pages[page["page"] - 1]
                    area = max(float(raw.width) * float(raw.height), 1.0)
                    page["tables"] = sum(
                        1 for table in raw.find_tables() if _table_shape_ok(table, area)
                    )
                    scanned += 1
                except Exception:
                    page["tables"] = 0
    except Exception as exc:
        log.warning("ruled table scan failed: %s", exc)
    return scanned


def _pdf_pages(path: Path) -> list[dict]:
    try:
        pages = _pdf_with_pdfium(path)
        renderer = "pdfium"
    except ImportError:
        pages = _pdf_with_pdfminer(path)
        renderer = "pdfminer"
    except Exception as exc:
        raise ValueError(f"PDF 解析失败：{exc}") from exc

    if not pages:
        raise ValueError("未能从该 PDF 中提取任何页面，可能文件为空或已损坏")
    for page in pages:
        page["column_ratio"] = _column_ratio(page["text"])
    scan_ruled_tables(path, pages)
    log.debug("pdf parsed with %s: %d pages", renderer, len(pages))
    return pages


# -------------------------------------------------------------- docx


def _md_table(rows: list[list[str]], caption: str = "") -> str:
    rows = [[re.sub(r"\s+", " ", str(cell or "")).strip().replace("|", "／") for cell in row] for row in rows]
    rows = [row for row in rows if any(row)]
    if len(rows) < 2:
        return ""
    width = max(len(row) for row in rows)
    rows = [row + [""] * (width - len(row)) for row in rows]
    header, body = rows[0], rows[1:]
    lines = []
    if caption:
        lines.append(caption)
    lines.append("| " + " | ".join(cell or f"列{i + 1}" for i, cell in enumerate(header)) + " |")
    lines.append("| " + " | ".join("---" for _ in header) + " |")
    lines.extend("| " + " | ".join(row) + " |" for row in body)
    return "\n".join(lines)


def _has_page_break(element) -> bool:
    """True when a paragraph carries a manual page break (w:br type=page)."""
    for node in element.iter():
        if isinstance(node.tag, str) and node.tag.endswith("}br"):
            if node.get("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}type") == "page":
                return True
    return False


def _docx_blocks(path: Path) -> list[str]:
    from docx import Document
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    document = Document(str(path))
    blocks: list[str] = []
    for child in document.element.body.iterchildren():
        tag = child.tag.split("}")[-1]
        if tag == "p":
            paragraph = Paragraph(child, document)
            text = paragraph.text.strip()
            if _has_page_break(child) and blocks and blocks[-1] != "\f":
                blocks.append("\f")
            if text:
                style = (paragraph.style.name or "").lower()
                if style.startswith("heading") or style.startswith("标题"):
                    blocks.append(f"# {text}")
                else:
                    blocks.append(text)
        elif tag == "tbl":
            table = Table(child, document)
            rendered = _md_table([[cell.text for cell in row.cells] for row in table.rows])
            if rendered:
                blocks.append(rendered)
    return blocks


def _native_pages(blocks: list[str], size: int) -> list[dict]:
    """Pack text blocks into virtual pages, honouring explicit form feeds."""
    pages: list[dict] = []
    buffer: list[str] = []
    length = 0

    def flush() -> None:
        nonlocal buffer, length
        if buffer:
            pages.append(_page_dict(len(pages) + 1, "\n\n".join(buffer)))
        buffer, length = [], 0

    for block in blocks:
        if block == "\f":
            flush()
            continue
        if length and length + len(block) > size:
            flush()
        buffer.append(block)
        length += len(block) + 2
    flush()
    return pages


def _docx_pages(path: Path) -> list[dict]:
    pages = _native_pages(_docx_blocks(path), settings.native_page_chars)
    if not pages:
        raise ValueError("Word 文档没有可提取的正文")
    return pages


# -------------------------------------------------------------- pptx


def _pptx_shape_lines(shape, depth: int = 0) -> list[str]:
    """Text of one shape, recursing into groups so flowcharts survive."""
    from pptx.enum.shapes import MSO_SHAPE_TYPE

    indent = "  " * depth
    lines: list[str] = []
    if shape.shape_type == MSO_SHAPE_TYPE.GROUP:
        for child in shape.shapes:
            lines.extend(_pptx_shape_lines(child, depth + 1))
        return lines

    if getattr(shape, "has_table", False) and shape.has_table:
        rows = [[cell.text for cell in row.cells] for row in shape.table.rows]
        rendered = _md_table(rows)
        if rendered:
            lines.append(rendered)
        return lines

    if getattr(shape, "has_text_frame", False) and shape.has_text_frame:
        for paragraph in shape.text_frame.paragraphs:
            text = "".join(run.text for run in paragraph.runs).strip()
            if not text:
                continue
            level = int(getattr(paragraph, "level", 0) or 0)
            marker = "• " if level else ""
            lines.append(f"{indent}{'  ' * level}{marker}{text}")
    return lines


def _pptx_pages(path: Path) -> list[dict]:
    from pptx import Presentation
    from pptx.enum.shapes import MSO_SHAPE_TYPE

    presentation = Presentation(str(path))
    pages: list[dict] = []
    for index, slide in enumerate(presentation.slides, start=1):
        blocks: list[str] = []
        pictures = 0
        nodes: list[tuple[str, float, float]] = []

        def collect(shape, depth: int = 0) -> None:
            nonlocal pictures
            if shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
                pictures += 1
            if shape.shape_type == MSO_SHAPE_TYPE.GROUP:
                for child in shape.shapes:
                    collect(child, depth + 1)
                return
            label = ""
            if getattr(shape, "has_text_frame", False) and shape.has_text_frame:
                label = shape.text_frame.text.strip()
            if label:
                try:
                    nodes.append(
                        (
                            label,
                            float(shape.left + shape.width / 2),
                            float(shape.top + shape.height / 2),
                        )
                    )
                except Exception:
                    nodes.append((label, 0.0, 0.0))
            blocks.extend(_pptx_shape_lines(shape, depth))

        for shape in slide.shapes:
            collect(shape)

        edges = _pptx_edges(slide, nodes)
        if edges:
            blocks.append("[连接关系]\n" + "\n".join(edges))

        notes = ""
        try:
            if slide.has_notes_slide:
                notes = (slide.notes_slide.notes_text_frame.text or "").strip()
        except Exception:
            notes = ""
        if notes:
            blocks.append(f"[备注] {notes}")

        # A slide is one page: page-level citations then read "p.3" = slide 3.
        item = _page_dict(index, "\n".join(block for block in blocks if block))
        item["image_cover"] = 0.5 if pictures and item["chars"] < 200 else 0.0
        item["vector_ops"] = 0
        pages.append(item)
    if not pages:
        raise ValueError("演示文稿没有可提取的幻灯片")
    return pages


def _pptx_edges(slide, nodes: list[tuple[str, float, float]]) -> list[str]:
    """Turn connectors into `A -> B` lines by matching endpoints to shapes."""
    if not nodes:
        return []

    def nearest(x: int, y: int) -> str:
        best, best_distance = "", None
        for label, cx, cy in nodes:
            distance = (cx - x) ** 2 + (cy - y) ** 2
            if best_distance is None or distance < best_distance:
                best, best_distance = label, distance
        return best

    edges: list[str] = []
    for shape in slide.shapes:
        if not hasattr(shape, "begin_x"):
            continue
        try:
            start = nearest(int(shape.begin_x), int(shape.begin_y))
            end = nearest(int(shape.end_x), int(shape.end_y))
        except Exception:
            continue
        if not start or not end or start == end:
            continue
        line = f"{start} -> {end}"
        if line not in edges:
            edges.append(line)
    return edges


# ------------------------------------------------------- xlsx / csv


def _rows_to_pages(blocks: list[tuple[str, list[list[str]]]], size: int) -> list[dict]:
    """One sheet may become several virtual pages, header repeated."""
    pages: list[dict] = []
    for sheet_name, rows in blocks:
        rows = [[str(cell or "") for cell in row] for row in rows]
        rows = [row for row in rows if any(cell.strip() for cell in row)]
        if not rows:
            continue
        header = rows[0]
        body = rows[1:] or [[""] * len(header)]
        chunk: list[list[str]] = []
        length = 0
        for row in body:
            row_len = sum(len(cell) for cell in row) + 4
            if chunk and length + row_len > size:
                pages.append(_page_dict(len(pages) + 1, _md_table([header] + chunk, f"## {sheet_name}")))
                chunk, length = [], 0
            chunk.append(row)
            length += row_len
        if chunk:
            pages.append(_page_dict(len(pages) + 1, _md_table([header] + chunk, f"## {sheet_name}")))
    return pages


def _xlsx_pages(path: Path) -> list[dict]:
    from openpyxl import load_workbook

    workbook = load_workbook(str(path), read_only=True, data_only=True)
    blocks: list[tuple[str, list[list[str]]]] = []
    try:
        for sheet in workbook.worksheets:
            rows: list[list[str]] = []
            for row in sheet.iter_rows(values_only=True):
                cells = ["" if cell is None else str(cell) for cell in row]
                if any(cell.strip() for cell in cells):
                    rows.append(cells)
                if len(rows) > 5000:  # a runaway sheet should not eat the process
                    break
            if rows:
                blocks.append((sheet.title, rows))
    finally:
        workbook.close()
    pages = _rows_to_pages(blocks, settings.native_page_chars)
    if not pages:
        raise ValueError("Excel 没有可提取的单元格内容")
    return pages


def _csv_pages(path: Path) -> list[dict]:
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError("CSV 编码无法识别（尝试过 utf-8 / gb18030）")

    rows = [row for row in csv.reader(io.StringIO(text)) if any(cell.strip() for cell in row)]
    pages = _rows_to_pages([(path.stem, rows)], settings.native_page_chars)
    if not pages:
        raise ValueError("CSV 没有可提取的内容")
    return pages


# ------------------------------------------------------ text / html


def _text_blocks(path: Path) -> list[str]:
    suffix = path.suffix.lower()
    if suffix in (".html", ".htm"):
        raw = path.read_bytes()
        for encoding in ("utf-8", "utf-8-sig", "gb18030"):
            try:
                html = raw.decode(encoding)
                break
            except UnicodeDecodeError:
                continue
        else:
            html = raw.decode("utf-8", errors="replace")
        html = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", html)
        html = re.sub(r"(?i)<br\s*/?>", "\n", html)
        html = re.sub(r"(?i)</(p|div|h[1-6]|li|tr)>", "\n", html)
        text = _HTML_TAG.sub(" ", html)
        text = re.sub(r"&nbsp;?", " ", text)
        text = re.sub(r"&amp;?", "&", text)
        text = re.sub(r"&lt;?", "<", text)
        text = re.sub(r"&gt;?", ">", text)
        parts = [line.strip() for line in text.split("\n")]
    else:
        raw = path.read_bytes()
        for encoding in ("utf-8-sig", "utf-8", "gb18030"):
            try:
                text = raw.decode(encoding)
                break
            except UnicodeDecodeError:
                continue
        else:
            text = raw.decode("utf-8", errors="replace")
        parts = text.split("\n")

    # Blank lines separate blocks: a Markdown heading, a paragraph, a list.
    blocks, buffer = [], []
    for line in parts:
        if line.strip():
            buffer.append(line.rstrip())
        elif buffer:
            blocks.append("\n".join(buffer))
            buffer = []
    if buffer:
        blocks.append("\n".join(buffer))
    return blocks


def _text_pages(path: Path) -> list[dict]:
    pages = _native_pages(_text_blocks(path), settings.native_page_chars)
    if not pages:
        raise ValueError("文件没有可提取的文本")
    return pages


# --------------------------------------------------------------- api


def extract_pages(path: str | Path) -> list[dict]:
    """Parse any supported file into page dicts. Raises ValueError."""
    path = Path(path)
    if not path.exists():
        raise ValueError(f"文件不存在：{path}")
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_EXT:
        raise ValueError(
            f"不支持的格式 {suffix or '（无扩展名）'}，"
            f"可用：{'、'.join(sorted(set(SUPPORTED_EXT)))}"
        )

    try:
        if suffix == ".pdf":
            return _pdf_pages(path)
        if suffix == ".docx":
            return _docx_pages(path)
        if suffix == ".pptx":
            return _pptx_pages(path)
        if suffix == ".xlsx":
            return _xlsx_pages(path)
        if suffix == ".csv":
            return _csv_pages(path)
        return _text_pages(path)
    except ImportError as exc:
        raise ValueError(f"缺少解析 {suffix} 所需的依赖：{exc}") from exc
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"解析失败：{exc}") from exc
