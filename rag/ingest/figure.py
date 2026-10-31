"""Diagram detection on top of loader.page_stats signals."""

import logging
from dataclasses import dataclass

log = logging.getLogger(__name__)

IMAGE_COVER_MIN = 0.05
FIGURE_OPS_MIN = 40


@dataclass
class FigureHint:
    page: int
    images: int
    image_cover: float
    vector_ops: int
    figure_like: bool
    reason: str


def figure_reason(page: dict) -> str:
    """Explain why a page is, or is not, treated as a diagram."""
    cover = float(page.get("image_cover", 0.0))
    ops = int(page.get("vector_ops", 0))
    chars = int(page.get("chars", 0))

    if page.get("scan_like"):
        return "扫描页，走 OCR 通道"
    if page.get("table_like"):
        return "表格页，走表格通道"
    if cover >= IMAGE_COVER_MIN:
        return f"嵌入图片覆盖 {cover:.0%}"
    if ops >= FIGURE_OPS_MIN:
        if chars >= 1500:
            return f"图元 {ops} 个但正文 {chars} 字，判为版式而非图形"
        return f"矢量图元 {ops} 个、正文仅 {chars} 字"
    return ""


def hints_from_pages(pages: list[dict]) -> dict[int, FigureHint]:
    """Build the per-page hint map from signals already stored on the pages."""
    hints: dict[int, FigureHint] = {}
    for page in pages:
        hints[page["page"]] = FigureHint(
            page=page["page"],
            images=1 if page.get("image_cover", 0) >= IMAGE_COVER_MIN else 0,
            image_cover=float(page.get("image_cover", 0.0)),
            vector_ops=int(page.get("vector_ops", 0)),
            figure_like=bool(page.get("figure_like")),
            reason=figure_reason(page) if page.get("figure_like") else "",
        )
    return hints
