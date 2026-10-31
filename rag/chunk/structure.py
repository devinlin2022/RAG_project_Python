"""Heading hierarchy for Chinese contracts and regulations. Regex only."""

import re
from dataclasses import dataclass

from rag.chunk.types import Section

_CN_NUM = "一二三四五六七八九十百千零〇两"

_PATTERNS: list[tuple[int, re.Pattern]] = [
    (1, re.compile(r"^(#{1})\s+(?=\S)")),
    (1, re.compile(rf"^(第\s*[{_CN_NUM}\d]+\s*(?:章|篇|部分))")),
    (1, re.compile(r"^(附\s*(?:件|录)\s*[一二三四五六七八九十\d]*)")),
    (2, re.compile(r"^(#{2})\s+(?=\S)")),
    (2, re.compile(rf"^(第\s*[{_CN_NUM}\d]+\s*节)")),
    (3, re.compile(r"^(#{3})\s+(?=\S)")),
    (3, re.compile(rf"^(第\s*[{_CN_NUM}\d]+\s*条)")),
    (4, re.compile(r"^(#{4})\s+(?=\S)")),
    (4, re.compile(rf"^([{_CN_NUM}]+\s*[、．.])")),
    (5, re.compile(r"^(#{5,6})\s+(?=\S)")),
    (5, re.compile(r"^(\d+(?:\.\d+){1,3}\s*[、．.]?\s*)")),
    (6, re.compile(r"^([（(][一二三四五六七八九十]+[）)])")),
    (7, re.compile(r"^(\d+\s*[、．.]\s*)")),
    (8, re.compile(r"^([（(]\d+[）)])")),
    (9, re.compile(r"^([①②③④⑤⑥⑦⑧⑨⑩⑪⑫])")),
]

_TITLE_STOP = "。；："
_TITLE_MAX = 30
_LINE_MAX = 200
_TABLE_ROW = re.compile(r"^\|.*\|")
_DECIMAL = re.compile(r"^(\d+(?:\.\d+)+)")


def _decimal_level(prefix: str) -> int:
    """2.1 -> 5, 2.1.3 -> 6, so deeper numbering nests instead of replacing."""
    match = _DECIMAL.match(prefix.strip())
    if not match:
        return 5
    return 5 + max(match.group(1).count(".") - 1, 0)


def parse_heading(line: str) -> tuple[int, str] | None:
    """Return (level, label) when the line starts a section."""
    text = line.strip()
    if not text or len(text) > _LINE_MAX:
        return None
    if _TABLE_ROW.match(text):
        return None

    for level, pattern in _PATTERNS:
        match = pattern.match(text)
        if not match:
            continue
        prefix = match.group(1)
        rest = text[match.end() :].strip()
        if prefix.startswith("#"):
            title = rest[:_TITLE_MAX].strip()
            return level, title or prefix

        if _DECIMAL.match(prefix.strip()):
            level = _decimal_level(prefix)

        label = re.sub(r"\s+", " ", prefix).strip()
        title = ""
        if rest:
            stop = min((rest.find(ch) for ch in _TITLE_STOP if ch in rest), default=-1)
            head = rest[:stop] if stop > 0 else rest
            head = head.strip()
            # A long remainder means the line already carries body text.
            if 0 < len(head) <= _TITLE_MAX:
                title = head
        return level, f"{label} {title}".strip()
    return None


def build_sections(pairs: list[tuple[int, object]]) -> list[Section]:
    """Turn ordered (page_no, block) pairs into heading-delimited sections."""
    sections: list[Section] = []
    stack: list[tuple[int, str]] = []
    buffer: list[str] = []
    pages: list[int] = []

    def path_of(depth: int | None = None) -> str:
        items = stack if depth is None else stack[: max(depth, 0)]
        return " > ".join(label for _, label in items) or "（正文）"

    def flush(kind: str = "text") -> None:
        if not buffer:
            return
        body = "\n".join(buffer).strip()
        buffer.clear()
        if not body:
            return
        page_list = pages or [0]
        sections.append(
            Section(
                label=stack[-1][1] if stack else "",
                path=path_of(),
                level=stack[-1][0] if stack else 0,
                text=body,
                page_from=min(page_list),
                page_to=max(page_list),
                kind=kind,
                parent=path_of(len(stack) - 1),
            )
        )
        pages.clear()

    for page_no, block in pairs:
        kind = getattr(block, "type", "text")
        text = getattr(block, "text", "")

        if kind in ("table", "figure"):
            flush()
            sections.append(
                Section(
                    label="",
                    path=path_of(),
                    level=stack[-1][0] if stack else 0,
                    text=text.strip(),
                    page_from=page_no,
                    page_to=page_no,
                    kind=kind,
                    parent=path_of(len(stack)),
                )
            )
            continue

        if kind != "text":
            # OCR text has no line structure; keep it whole in the current section.
            pages.append(page_no)
            buffer.append(text)
            continue

        for line in text.splitlines():
            heading = parse_heading(line)
            if heading:
                flush()
                level, label = heading
                while stack and stack[-1][0] >= level:
                    stack.pop()
                stack.append((level, label))
                pages.append(page_no)
                buffer.append(line.strip())
            elif line.strip():
                pages.append(page_no)
                buffer.append(line.rstrip())
    flush()
    return sections


@dataclass
class HeadingStats:
    heading_lines: int
    total_lines: int
    max_depth: int
    article_count: int
    styles: dict[str, int]

    @property
    def ratio(self) -> float:
        return self.heading_lines / max(self.total_lines, 1)


def heading_stats(text: str) -> HeadingStats:
    """Cheap structural census used by the strategy recommender."""
    lines = [line for line in text.splitlines() if line.strip()]
    hits = 0
    depth = 0
    articles = 0
    styles: dict[str, int] = {}

    for line in lines:
        heading = parse_heading(line)
        if not heading:
            continue
        hits += 1
        level, _ = heading
        depth = max(depth, level)
        style = _style_of(line)
        styles[style] = styles.get(style, 0) + 1
        if re.match(rf"^第\s*[{_CN_NUM}\d]+\s*条", line.strip()):
            articles += 1

    return HeadingStats(hits, len(lines), depth, articles, styles)


def _style_of(line: str) -> str:
    text = line.strip()
    if text.startswith("#"):
        return "markdown"
    if re.match(rf"^第\s*[{_CN_NUM}\d]+\s*条", text):
        return "条"
    if re.match(rf"^第\s*[{_CN_NUM}\d]+\s*(章|篇|部分)", text):
        return "章"
    if re.match(rf"^第\s*[{_CN_NUM}\d]+\s*节", text):
        return "节"
    if re.match(rf"^[{_CN_NUM}]+\s*[、．.]", text):
        return "一、"
    if re.match(r"^[（(][一二三四五六七八九十]+[）)]", text):
        return "（一）"
    if re.match(r"^\d+(?:\.\d+)+\s*", text):
        return "1.1"
    if re.match(r"^\d+\s*[、．.]\s*", text):
        return "1."
    if re.match(r"^[（(]\d+[）)]", text):
        return "（1）"
    if re.match(r"^[①-⑫]", text):
        return "①"
    return "其他"
