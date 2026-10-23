"""Route each page to the rails it needs, then run only those rails."""

import logging
import re
import time
from dataclasses import asdict, dataclass, field

from rag.config import settings
from rag.ingest.figure import hints_from_pages, figure_reason
from rag.ingest.ocr import OcrError, describe_figure, ocr_local, ocr_page
from rag.ingest.render import render_page
from rag.ingest.table import tables_to_markdown

log = logging.getLogger(__name__)

# OCR text this short on a fully covered page means the page is a picture.
PICTURE_TEXT_MAX = 120
PICTURE_COVER_MIN = 0.6


@dataclass
class Block:
    type: str  # text | table | figure | ocr
    text: str
    source: str
    ms: int = 0


@dataclass
class PagePlan:
    page: int
    chars: int
    scan_like: bool
    table_like: bool
    figure_like: bool
    routes: list[str]
    reason: str


@dataclass
class PageResult:
    page: int
    routes: list[str]
    status: str  # done | skipped | error
    chars_before: int
    chars_after: int
    ms: int
    engine: str = ""
    error: str = ""
    blocks: list[Block] = field(default_factory=list)


def plan_pages(doc: dict) -> list[PagePlan]:
    """Decide routes per page. Pure function: no network, no rendering."""
    plans: list[PagePlan] = []
    for page in doc["pages"]:
        scan_like = bool(page["scan_like"])
        table_like = bool(page["table_like"])
        figure_like = bool(page["figure_like"])

        routes: list[str] = []
        if scan_like:
            routes.append("ocr")
            reason = (
                f"正文 {page['chars']} 字、图片覆盖 {page.get('image_cover', 0):.0%}，"
                "内容不在文本层"
            )
        else:
            routes.append("text")
            reason = f"文本层可用（{page['chars']} 字符）"
        if table_like:
            routes.append("table")
            reason += f"；检测到 {page['tables']} 个表格"
        if figure_like:
            routes.append("figure")
            detail = figure_reason(page)
            reason += f"；{detail}" if detail else "；判定为图形页"

        plans.append(
            PagePlan(
                page=page["page"],
                chars=page["chars"],
                scan_like=scan_like,
                table_like=table_like,
                figure_like=figure_like,
                routes=routes,
                reason=reason,
            )
        )
    return plans


def _strip_covered_lines(text: str, tables: list[str]) -> str:
    """Drop text lines that a table block already contains, keep everything else."""
    flat = re.sub(r"\s+", "", " ".join(tables))
    kept = [
        line
        for line in text.splitlines()
        if re.sub(r"\s+", "", line) and re.sub(r"\s+", "", line) not in flat
    ]
    return "\n".join(kept).strip()


def _budget_pages(plans: list[PagePlan], limit: int) -> set[int]:
    """Pages allowed to call a vision model, in document order."""
    candidates = [p.page for p in plans if "ocr" in p.routes or "figure" in p.routes]
    return set(candidates[:limit]) if limit > 0 else set(candidates)


def enhance(
    doc: dict,
    limit_pages: int | None = None,
    on_event=None,
) -> dict:
    """Run the enhancement rails and return a report. Blocks are stored on the doc."""
    started = time.perf_counter()
    emit = on_event or (lambda event: None)

    hints = hints_from_pages(doc["pages"])
    plans = plan_pages(doc)
    limit = settings.ocr_max_pages if limit_pages is None else limit_pages
    allowed = _budget_pages(plans, limit)
    vision_off = not settings.ocr_ready

    emit({"stage": "plan", "plans": [asdict(p) for p in plans]})

    results: list[PageResult] = []
    api_calls = 0
    skipped: list[int] = []
    total_started = time.perf_counter()

    for plan in plans:
        page_started = time.perf_counter()
        page = doc["pages"][plan.page - 1]
        blocks: list[Block] = []

        needs_api = "ocr" in plan.routes or "figure" in plan.routes
        if needs_api and (vision_off or plan.page not in allowed):
            reason = (
                "OCR 未启用：设置 OCR_ENABLED=true 并配置 Key 后重启"
                if vision_off
                else f"超出 OCR 页数预算（OCR_MAX_PAGES={limit}）"
            )
            skipped.append(plan.page)
            results.append(
                PageResult(
                    page=plan.page,
                    routes=plan.routes,
                    status="skipped",
                    chars_before=plan.chars,
                    chars_after=plan.chars,
                    ms=0,
                    error=reason,
                    blocks=[Block("text", page["text"], "pdfminer", 0)],
                )
            )
            emit({"stage": "skip", "page": plan.page, "reason": reason})
            continue

        if needs_api and (time.perf_counter() - total_started) > settings.enhance_max_seconds:
            skipped.append(plan.page)
            results.append(
                PageResult(
                    page=plan.page,
                    routes=plan.routes,
                    status="skipped",
                    chars_before=plan.chars,
                    chars_after=plan.chars,
                    ms=0,
                    error=f"超出时间预算（ENHANCE_MAX_SECONDS={settings.enhance_max_seconds}）",
                )
            )
            emit({"stage": "skip", "page": plan.page, "reason": "超出时间预算"})
            continue

        emit({"stage": "page_start", "page": plan.page, "routes": plan.routes})
        error = ""
        png: bytes | None = None

        try:
            if "ocr" in plan.routes:
                png = render_page(doc["path"], plan.page, settings.ocr_render_dpi)
                result = ocr_page(png)
                api_calls += 1
                if result.empty:
                    raise OcrError("视觉模型未返回任何文字")
                blocks.append(Block("ocr", result.text, result.engine, result.ms))
                emit({"stage": "route_done", "page": plan.page, "route": "ocr",
                      "ms": result.ms, "chars": len(result.text)})
            else:
                blocks.append(Block("text", page["text"], "pdfminer", 0))

            if "table" in plan.routes:
                table_started = time.perf_counter()
                tables = tables_to_markdown(doc["path"], plan.page, page_text=page["text"])
                table_ms = int((time.perf_counter() - table_started) * 1000)
                for item in tables:
                    blocks.append(Block("table", item, "pdfplumber", table_ms))

                # Drop only lines the table already covers, so prose survives.
                text_block = next((b for b in blocks if b.type == "text"), None)
                table_chars = sum(len(item) for item in tables)
                replaced = False
                if text_block and tables:
                    remaining = _strip_covered_lines(text_block.text, tables)
                    replaced = remaining != text_block.text
                    if remaining.strip():
                        text_block.text = remaining
                    else:
                        blocks.remove(text_block)

                emit({"stage": "route_done", "page": plan.page, "route": "table",
                      "ms": table_ms, "chars": table_chars, "replaced_text": replaced})

            hint = hints.get(plan.page)
            cover = hint.image_cover if hint else 0.0
            if "figure" in plan.routes:
                # The planner already judged this page to be a diagram.
                wants_figure = True
            elif "ocr" in plan.routes and cover >= PICTURE_COVER_MIN:
                # Coverage cannot separate a scanned page from a picture; OCR volume can.
                ocr_text = next((b.text for b in blocks if b.type == "ocr"), "")
                wants_figure = len(ocr_text) < PICTURE_TEXT_MAX
            else:
                wants_figure = False
            if wants_figure:
                png = png or render_page(doc["path"], plan.page, settings.ocr_render_dpi)
                result = describe_figure(png)
                api_calls += 1
                if result.text:
                    label = f"[第 {plan.page} 页图注] "
                    blocks.append(Block("figure", label + result.text, result.engine, result.ms))
                    emit({"stage": "route_done", "page": plan.page, "route": "figure",
                          "ms": result.ms, "chars": len(result.text)})
        except (OcrError, ValueError) as exc:
            error = str(exc)
            log.warning("page %s enhancement failed: %s", plan.page, exc)
            if not blocks:
                blocks.append(Block("text", page["text"], "pdfminer", 0))
            emit({"stage": "route_error", "page": plan.page, "error": error})

        chars_after = sum(len(b.text) for b in blocks)
        results.append(
            PageResult(
                page=plan.page,
                routes=plan.routes,
                status="error" if error else "done",
                chars_before=plan.chars,
                chars_after=chars_after,
                ms=int((time.perf_counter() - page_started) * 1000),
                engine=next((b.source for b in blocks if b.type in ("ocr", "figure")), ""),
                error=error,
                blocks=blocks,
            )
        )
        emit({"stage": "page_done", "page": plan.page, "ms": results[-1].ms,
              "chars": chars_after, "error": error})

    report = {
        "pages": results,
        "plans": plans,
        "hints": hints,
        "ms": int((time.perf_counter() - started) * 1000),
        "api_calls": api_calls,
        "skipped": skipped,
        "ocr_ready": settings.ocr_ready,
        "ocr_model": settings.ocr_model,
        "vision_model": settings.vision_model,
        "text_before": sum(r.chars_before for r in results),
        "text_after": sum(r.chars_after for r in results),
    }
    doc["enhance"] = report
    emit({"stage": "done", "ms": report["ms"], "api_calls": api_calls})
    return report


def blocks_of(doc: dict, page_no: int) -> list[Block]:
    """Blocks for one page: enhanced when the report exists, else the text layer."""
    report = doc.get("enhance")
    if report:
        for result in report["pages"]:
            if result.page == page_no and result.blocks:
                return result.blocks
    for page in doc["pages"]:
        if page["page"] == page_no:
            return [Block("text", page["text"], "pdfminer", 0)]
    return []


def all_blocks(doc: dict) -> list[tuple[int, Block]]:
    """Ordered (page_no, block) pairs for the whole document."""
    pairs: list[tuple[int, Block]] = []
    for page in doc["pages"]:
        for block in blocks_of(doc, page["page"]):
            pairs.append((page["page"], block))
    return pairs


def to_local_ocr(png: bytes):
    """Expose the offline backend without importing it in the UI layer."""
    return ocr_local(png)
