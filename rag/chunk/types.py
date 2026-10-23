"""Chunk and section types shared by every splitter."""

from dataclasses import dataclass, field


@dataclass
class Chunk:
    idx: int
    text: str
    page_from: int
    page_to: int
    kind: str = "text"  # text | table | figure | ocr
    section: str = ""  # heading path, e.g. "第一章 总则 > 第二条 适用范围"
    level: int = 0
    strategy: str = ""
    merged_from: int = 1  # how many source sections were merged into this chunk

    @property
    def chars(self) -> int:
        return len(self.text)

    def to_dict(self) -> dict:
        return {
            "idx": self.idx,
            "text": self.text,
            "page_from": self.page_from,
            "page_to": self.page_to,
            "kind": self.kind,
            "section": self.section,
            "level": self.level,
            "strategy": self.strategy,
            "chars": self.chars,
        }


@dataclass
class Section:
    """A heading-delimited unit. May span pages."""

    label: str
    path: str
    level: int
    text: str
    page_from: int
    page_to: int
    kind: str = "text"
    parent: str = ""
    children: list[str] = field(default_factory=list)

    @property
    def chars(self) -> int:
        return len(self.text)
