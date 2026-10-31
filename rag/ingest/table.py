"""Turn a page's tables into Markdown.

Ruled grids come from pdfplumber; pipe text and space-aligned text are parsed
from the extracted text.
"""

import logging
import re
from collections import Counter

log = logging.getLogger(__name__)

_HEADER_HINTS = ("项目", "名称", "序号", "条款", "内容", "金额", "说明", "单位", "数量", "标准", "备注")
_SPACE_SPLIT = re.compile(r"\s{2,}")


def _clean_cell(value) -> str:
    if value is None:
        return ""
    text = re.sub(r"\s+", " ", str(value)).strip()
    return text.replace("|", "／")[:300]


def _looks_like_header(row: list[str]) -> bool:
    cells = [c for c in row if c]
    if len(cells) < 2:
        return False
    if any(hint in " ".join(cells) for hint in _HEADER_HINTS):
        return True
    # A header row holds short labels, never bare numbers.
    if any(cell.replace(".", "").replace("-", "").isdigit() for cell in cells):
        return False
    return all(len(cell) <= 24 and len(cell.split()) <= 3 for cell in cells)


def _render(rows: list[list[str]], caption: str) -> str | None:
    rows = [[_clean_cell(c) for c in row] for row in rows]
    rows = [r for r in rows if any(r)]
    if len(rows) < 2:
        return None

    width = max(len(r) for r in rows)
    if width < 2:
        return None
    # Layout blocks are 2x2 grids too; real cells are short, layout cells hold prose.
    if len(rows) < 3 and width < 3:
        return None
    cell_lengths = sorted(len(c) for row in rows for c in row if c)
    if cell_lengths:
        median = cell_lengths[len(cell_lengths) // 2]
        if median > 80:
            return None
    rows = [r + [""] * (width - len(r)) for r in rows]

    # A header shorter than the table means a trailing empty column, not a row.
    if _looks_like_header(rows[0]) or len(rows[0]) < width:
        header, body = rows[0] + [""] * (width - len(rows[0])), rows[1:]
    else:
        header, body = [f"列{i + 1}" for i in range(width)], rows

    body = [r for r in body if r != header]
    lines = [
        caption,
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    lines += ["| " + " | ".join(r) + " |" for r in body]
    return "\n".join(lines)


def _pipe_cells(line: str) -> list[str] | None:
    text = line.strip()
    if text.count("|") < 2:
        return None
    inner = text.strip("|")
    cells = [cell.strip() for cell in inner.split("|")]
    cells = [c for c in cells if c]
    if len(cells) < 3:
        return None
    if len(" ".join(cells)) < 6:
        return None
    return cells


def _space_cells(line: str) -> list[str] | None:
    text = line.rstrip()
    if len(text) < 12:
        return None
    cells = [cell.strip() for cell in _SPACE_SPLIT.split(text) if cell.strip()]
    if len(cells) < 3:
        return None
    return cells


def text_tables(text: str, caption: str = "[表 · 文本对齐]") -> list[str]:
    """Parse pipe-delimited or space-aligned table runs out of plain page text."""
    lines = text.splitlines()

    for extractor, min_rows in ((_pipe_cells, 2), (_space_cells, 3)):
        blocks: list[str] = []
        run: list[list[str]] = []
        for line in lines + [""]:  # sentinel flushes the last run
            cells = extractor(line)
            if cells is None:
                if len(run) >= min_rows:
                    counts = Counter(len(r) for r in run)
                    if counts.most_common(1)[0][1] >= min_rows:
                        rendered = _render(run, caption)
                        if rendered:
                            blocks.append(rendered)
                run = []
            else:
                run.append(cells)
        if blocks:
            return blocks
    return []


def tables_to_markdown(path: str, page_no: int, page_text: str = "") -> list[str]:
    """Ruled tables first, then text-parsed tables. Never raises."""
    blocks: list[str] = []
    try:
        import pdfplumber
    except ImportError:
        return text_tables(page_text, f"[表 · 第 {page_no} 页]") if page_text else []

    try:
        with pdfplumber.open(str(path)) as pdf:
            if 1 <= page_no <= len(pdf.pages):
                for index, table in enumerate(pdf.pages[page_no - 1].find_tables(), start=1):
                    try:
                        rendered = _render(table.extract(), f"[表 {index} · 第 {page_no} 页]")
                    except Exception:
                        continue
                    if rendered:
                        blocks.append(rendered)
    except Exception as exc:
        log.warning("ruled table extraction failed on page %s: %s", page_no, exc)

    if blocks:
        return blocks
    if page_text:
        return text_tables(page_text, f"[表 · 第 {page_no} 页]")
    return []


def split_markdown_table(block: str, max_chars: int) -> list[str]:
    """Split a Markdown table by rows, repeating header on each fragment."""
    lines = block.splitlines()
    if len(lines) < 4:
        return [block] if block.strip() else []

    caption = lines[0]
    header = lines[1]
    separator = lines[2]
    prefix = "\n".join([caption, header, separator])
    budget = max(max_chars - len(prefix) - 1, 80)

    fragments: list[str] = []
    current: list[str] = []
    size = 0
    for row in lines[3:]:
        if size + len(row) + 1 > budget and current:
            fragments.append(prefix + "\n" + "\n".join(current))
            current, size = [], 0
        current.append(row)
        size += len(row) + 1
    if current:
        fragments.append(prefix + "\n" + "\n".join(current))
    return fragments or [block]
