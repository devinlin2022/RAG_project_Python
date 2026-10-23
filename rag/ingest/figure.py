"""Detect pages whose content is a picture rather than text.

Two signals, both cheap and offline:

- image_cover: embedded raster area / page area. Filters out header logos (<5%).
- vector_ops: count of rects/lines/curves. A vector flowchart is drawn as hundreds
  of tiny shapes with almost no text, so ops >> text is the giveaway.

A table is also drawn with rects, so pages already known to hold tables are skipped.
"""

import logging
from dataclasses import dataclass

log = logging.getLogger(__name__)

IMAGE_COVER_MIN = 0.05
VECTOR_OPS_MIN = 60
VECTOR_TEXT_MAX = 1500


@dataclass
class FigureHint:
    page: int
    images: int
    image_cover: float
    vector_ops: int
    figure_like: bool
    reason: str


def detect_figures(
    path: str,
    skip_pages: set[int] | None = None,
) -> dict[int, FigureHint]:
    """Per-page figure hints. Returns {} when pdfplumber is unavailable."""
    skip_pages = skip_pages or set()
    hints: dict[int, FigureHint] = {}
    try:
        import pdfplumber
    except ImportError:
        return hints

    try:
        with pdfplumber.open(str(path)) as pdf:
            for index, page in enumerate(pdf.pages, start=1):
                try:
                    area = max(float(page.width) * float(page.height), 1.0)
                    images = page.images or []
                    cover = min(
                        sum(float(i["width"]) * float(i["height"]) for i in images) / area,
                        1.0,
                    )
                    ops = len(page.rects or []) + len(page.lines or []) + len(page.curves or [])
                    chars = len((page.extract_text() or "").strip())
                except Exception:
                    continue

                if index in skip_pages:
                    hints[index] = FigureHint(index, len(images), round(cover, 3), ops, False, "表格页，走表格通道")
                    continue

                if cover >= IMAGE_COVER_MIN:
                    ok, reason = True, f"嵌入图片覆盖 {cover:.0%}"
                elif ops >= VECTOR_OPS_MIN and chars < VECTOR_TEXT_MAX:
                    ok, reason = True, f"矢量图元 {ops} 个、正文仅 {chars} 字"
                else:
                    ok, reason = False, ""
                hints[index] = FigureHint(index, len(images), round(cover, 3), ops, ok, reason)
    except Exception as exc:
        log.warning("figure detection failed: %s", exc)
    return hints
