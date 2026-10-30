"""Gradio UI: a document workbench and a question box, in that order.

The workbench is six numbered steps (upload -> parse -> chunks -> pages ->
source -> metrics) laid out as a directory, because the whole pipeline on one
screen was unreadable. Rendering and event binding only: every handler
delegates to rag.service.
"""

import html
import logging
from pathlib import Path

import gradio as gr

from rag.config import VERSION, settings
from rag.ingest.extract import SUPPORTED_EXT
from rag.ingest.loader import page_text as page_text_view
from rag.service import Service, ServiceError
from rag.utils import format_pages

log = logging.getLogger(__name__)

DOC_HEADERS = ["ID", "文件名", "格式", "页", "字符", "块", "已增强", "状态"]
PAGE_HEADERS = ["页", "字符", "表格", "列结构", "图像", "扫描", "表格页", "图形页"]
SOURCE_HEADERS = ["#", "文件", "页", "类型", "融合分", "向量", "BM25", "精排", "预览"]
COMPARE_HEADERS = ["模式", "命中块", "覆盖页", "页覆盖率", "首位分", "首位优势", "精排", "耗时"]

MODE_CHOICES = [("快速检索", "fast"), ("增强检索", "enhanced")]

# Which pipeline steps each mode actually runs. Fast mode merges the two rails
# into one line on purpose: showing five rows for a 900 ms path reads as noise.
PLAN_STEPS = {
    "fast": ["embed", "retrieve", "generate"],
    "enhanced": ["rewrite", "embed", "dense", "sparse", "fusion", "rerank", "generate"],
}

STEPS = [
    ("upload", "① 上传建索引", "文件解析、切块、算向量"),
    ("parse", "② 解析结果", "格式识别与逐页文本量"),
    ("chunks", "③ 分块地图", "每块的位置、类型、字数"),
    ("pages", "④ 页级诊断", "每页的扫描 / 表格 / 图形信号"),
    ("text", "⑤ 原文查看", "逐页核对识别结果"),
    ("metrics", "⑥ 检索指标", "两种模式在这份文档上的实测表现"),
]
STEP_LABELS = {key: label for key, label, _ in STEPS}

KIND_STYLE = {
    "text": ("正文", "#e8eef7", "#c8d7ea"),
    "table": ("表格", "#e4f3e8", "#bfdcc8"),
    "figure": ("图注", "#fdeee0", "#f0cfae"),
    "ocr": ("OCR", "#f1e8fb", "#d8c3f0"),
}


def _update(**kwargs):
    updater = getattr(gr, "update", None)
    if updater is not None:
        return updater(**kwargs)
    return gr.Dropdown(choices=kwargs.get("choices", []), value=kwargs.get("value"))


def _file_paths(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        out: list[str] = []
        for item in value:
            out.extend(_file_paths(item))
        return out
    name = getattr(value, "name", None)
    return [name] if isinstance(name, str) else []


def _doc_choices(service: Service) -> list[tuple[str, str]]:
    return [
        (f"{row['filename']}（{row['pages']} 页 · {row['chunks']} 块）", row["id"])
        for row in service.list_documents()
    ]


def _doc_rows(service: Service) -> list[list]:
    return [
        [
            row["id"],
            row["filename"],
            row["format"],
            row["pages"],
            row["chars"],
            row["chunks"],
            "是" if row["enhanced"] else "",
            "已建索引" if row["indexed_at"] != "—" else "待建索引",
        ]
        for row in service.list_documents()
    ]


def _page_rows(doc: dict) -> tuple[list[list], list[int]]:
    rows, pages = [], []
    for page in doc["pages"]:
        rows.append(
            [
                page["page"],
                page["chars"],
                page["tables"],
                round(page.get("column_ratio", 0), 2),
                f"{page.get('image_cover', 0):.0%}",
                "是" if page["scan_like"] else "",
                "是" if page["table_like"] else "",
                "是" if page["figure_like"] else "",
            ]
        )
        pages.append(page["page"])
    return rows, pages


def _doc_summary(doc: dict) -> str:
    flags = []
    if doc["scan_like_pages"]:
        flags.append(f"扫描页 {len(doc['scan_like_pages'])}（{format_pages(doc['scan_like_pages'][:8])}）")
    if doc["table_like_pages"]:
        flags.append(f"表格页 {len(doc['table_like_pages'])}（{format_pages(doc['table_like_pages'][:8])}）")
    if doc["figure_like_pages"]:
        flags.append(f"图形页 {len(doc['figure_like_pages'])}（{format_pages(doc['figure_like_pages'][:8])}）")
    return (
        f"**{doc['filename']}**　｜　{doc.get('format', '')}　｜　{doc['page_count']} 页"
        f"　｜　{doc['chars']:,} 字符　｜　{doc.get('chunk_count', 0)} 块"
        f" / {doc.get('dim', 0)} 维\n\n"
        + ("- " + "　".join(flags) + "\n" if flags else "- 文本层完整，未发现扫描页或表格页\n")
        + f"- 增强识别：{'已完成' if doc.get('enhanced_at') else '未运行'}"
    )


# ------------------------------------------------------------------ visuals


def _bar(pct: float, color: str, border: str, height: int = 10) -> str:
    return (
        f"<span style='display:inline-block;height:{height}px;border-radius:3px;"
        f"background:{color};border:1px solid {border};width:{max(pct, 0.6):.2f}%'></span>"
    )


def flow_html(steps: list[dict]) -> str:
    """The six-step directory as a progress strip: where you are, what is ready.

    A tab bar tells you what exists; this tells you what has actually been done
    to the document, which is the part a reader cannot infer from a filename.
    """
    chips = []
    for index, step in enumerate(steps):
        done = step.get("done")
        color = "#1a7f37" if done else "#8b949e"
        mark = "✓" if done else str(index + 1)
        background = "#e8f5ec" if done else "#f6f8fa"
        chips.append(
            "<div style='display:flex;flex-direction:column;gap:2px;flex:1 1 0;min-width:96px;"
            f"padding:6px 8px;border:1px solid {'#bfdcc8' if done else '#d8dee4'};"
            f"border-radius:6px;background:{background}'>"
            f"<div style='font-size:10px;color:{color};letter-spacing:.04em'>"
            f"{mark}　{html.escape(step['label'])}</div>"
            f"<div style='font-size:11px;color:#57606a;white-space:nowrap;overflow:hidden;"
            f"text-overflow:ellipsis'>{html.escape(step.get('hint', ''))}</div>"
            "</div>"
        )
        if index < len(steps) - 1:
            chips.append(
                "<div style='flex:0 0 10px;align-self:center;color:#d0d7de;font-size:12px'>›</div>"
            )
    return (
        "<div style=\"display:flex;gap:4px;align-items:stretch;font-family:ui-sans-serif,"
        "system-ui,'Segoe UI',sans-serif;margin:2px 0 10px\">" + "".join(chips) + "</div>"
    )


def step_states(doc: dict | None, chunks: list) -> list[dict]:
    """Ordered steps with a done flag and a one-line hint each."""
    doc = doc or {}
    pages = doc.get("page_count", 0)
    scans = len(doc.get("scan_like_pages", []))
    hints = {
        "upload": (f"{doc.get('filename', '未选择文档')}", bool(doc)),
        "parse": (f"{doc.get('format', '—')} · {pages} 页 · {doc.get('chars', 0):,} 字", bool(pages)),
        "chunks": (f"{len(chunks)} 块" if chunks else "尚未建索引", bool(chunks)),
        "pages": (
            f"扫描页 {scans} · 表格页 {len(doc.get('table_like_pages', []))}" if doc else "选择文档后展开",
            bool(doc),
        ),
        "text": ("按页查看原文" if doc else "选择文档后展开", bool(doc)),
        "metrics": ("跑一次评测看两种模式", False),
    }
    return [
        {"key": key, "label": label, "hint": hints[key][0], "done": hints[key][1]}
        for key, label, _ in STEPS
    ]


def page_strip_html(doc: dict | None) -> str:
    """Per-page text volume as a strip, so a 40-page report shows its shape."""
    if not doc or not doc.get("pages"):
        return ""
    chars = [page["chars"] for page in doc["pages"]]
    peak = max(chars) or 1
    cells = []
    for page in doc["pages"]:
        shade = page["chars"] / peak
        background = f"rgba(31,111,235,{0.10 + 0.55 * shade:.2f})" if page["chars"] else "#eef1f4"
        flag = page.get("scan_like") and "扫描" or page.get("table_like") and "表格" or ""
        title = f"第 {page['page']} 页 · {page['chars']} 字符" + (f" · {flag}" if flag else "")
        cells.append(
            f"<div title='{title}' style=\"flex:1 1 0;min-width:4px;height:16px;"
            f"background:{background}\"></div>"
        )
    thin = sum(1 for value in chars if value < settings.scan_page_char_threshold)
    return (
        "<div style='display:flex;gap:2px;margin:6px 0 4px'>" + "".join(cells) + "</div>"
        "<div style='font-size:11px;color:#8b949e'>页轴：%d 页 · 颜色越深文字越多 · "
        "低于 %d 字的页 %d 页（会触发扫描件判断）</div>"
        % (len(chars), settings.scan_page_char_threshold, thin)
    )


def chunk_map_html(chunks: list, hits: list[dict] | None = None, limit: int = 240) -> str:
    """Blocks as proportional bars: what the splitter produced, and what answered."""
    if not chunks:
        return (
            "<div style='color:#57606a;font-size:13px;padding:8px 0'>"
            "还没有分块。上传文件并建立索引后，这里会显示每个块的位置与大小。</div>"
        )

    rank = {f"{item['doc_id']}:{item['page']}": item["n"] for item in (hits or [])}
    biggest = max(chunk.chars for chunk in chunks) or 1
    pages = max(chunk.page_to for chunk in chunks) or 1

    per_page: dict[int, int] = {}
    for chunk in chunks:
        for page in range(chunk.page_from, chunk.page_to + 1):
            per_page[page] = per_page.get(page, 0) + chunk.chars
    peak = max(per_page.values()) if per_page else 1
    cells = []
    for page in range(1, pages + 1):
        shade = per_page.get(page, 0) / peak
        background = f"rgba(31,111,235,{0.10 + 0.55 * shade:.2f})" if shade else "#eef1f4"
        ring = ""
        if hits:
            cited = any(
                item["page"] <= page <= item.get("page_to", item["page"]) for item in hits
            )
            ring = "box-shadow:0 0 0 1.5px #d1242f;" if cited else ""
        cells.append(
            f"<div title='第 {page} 页 · {per_page.get(page, 0)} 字符' style=\"flex:1 1 0;"
            f"min-width:4px;height:16px;background:{background};{ring}\"></div>"
        )

    rows = []
    for chunk in chunks[:limit]:
        hit = rank.get(f"{chunk.doc_id}:{chunk.page_from}")
        label, fill, edge = KIND_STYLE.get(chunk.kind, KIND_STYLE["text"])
        width = chunk.chars / biggest * 100
        badge = (
            f"<span style='background:#d1242f;color:#fff;border-radius:9px;padding:0 6px;"
            f"font-size:10px;margin-left:6px'>引用 {hit}</span>"
            if hit
            else ""
        )
        outline = "outline:1.5px solid #d1242f;outline-offset:1px;" if hit else ""
        preview = html.escape(" ".join(chunk.text.split())[:72])
        rows.append(
            "<div style='display:flex;align-items:center;gap:8px;padding:2px 0;"
            f"font-size:12px;color:#1f2328;{outline}'>"
            f"<span style='flex:0 0 46px;color:#57606a'>#{chunk.idx}</span>"
            f"<span style='flex:0 0 58px;color:#57606a'>"
            f"p.{chunk.page_from}{'' if chunk.page_from == chunk.page_to else '-' + str(chunk.page_to)}</span>"
            f"<span style='flex:0 0 34px;color:#57606a'>{label}</span>"
            f"<span style='flex:0 0 96px;text-align:right;color:#1f2328'>{chunk.chars} 字</span>"
            f"<span style='flex:0 0 34%;'>{_bar(width, fill, edge)}</span>"
            f"<span style='flex:1 1 auto;color:#57606a;overflow:hidden;"
            f"text-overflow:ellipsis;white-space:nowrap'>{preview}{badge}</span>"
            "</div>"
        )

    more = (
        f"<div style='color:#8b949e;font-size:11px;padding-top:4px'>"
        f"共 {len(chunks)} 块，仅显示前 {limit} 块</div>"
        if len(chunks) > limit
        else ""
    )

    legend = "".join(
        f"<span style='display:inline-block;width:10px;height:10px;background:{fill};"
        f"border:1px solid {edge};border-radius:2px;margin-right:4px'></span>"
        f"<span style='margin-right:14px;color:#57606a'>{name}</span>"
        for name, fill, edge in KIND_STYLE.values()
    )

    return (
        "<div style=\"font-family:ui-sans-serif,system-ui,'Segoe UI',sans-serif\">"
        "<div style='display:flex;gap:2px;margin-bottom:8px'>"
        + "".join(cells)
        + "</div>"
        "<div style='font-size:11px;color:#8b949e;margin-bottom:6px'>"
        f"页轴：{pages} 页 · 颜色越深文字越多"
        + ("　·　红框＝被引用页" if hits else "")
        + "</div>"
        "<div style='max-height:340px;overflow:auto;border:1px solid #d8dee4;border-radius:6px;"
        "padding:6px 8px;background:#fff'>"
        + "".join(rows)
        + more
        + "</div>"
        "<div style='font-size:11px;margin-top:8px'>"
        + legend
        + "</div></div>"
    )


def chunk_summary_md(chunks: list) -> str:
    if not chunks:
        return ""
    total = sum(chunk.chars for chunk in chunks)
    avg = total // len(chunks)
    kinds: dict[str, int] = {}
    for chunk in chunks:
        kinds[chunk.kind] = kinds.get(chunk.kind, 0) + 1
    breakdown = "　".join(
        f"{KIND_STYLE.get(kind, ('其他',))[0]} {count}" for kind, count in sorted(kinds.items())
    )
    spans = [chunk.page_to - chunk.page_from + 1 for chunk in chunks]
    return (
        f"共 **{len(chunks)}** 块，合计 {total:,} 字符，平均 {avg} 字/块，"
        f"单块最长跨 {max(spans)} 页。{breakdown}"
    )


STEP_TITLES = {
    "rewrite": ("理解问题", "把追问补全成可独立检索的问题"),
    "embed": ("向量化查询", "把问题编码成与块同一空间的向量"),
    "dense": ("向量检索", "余弦相似度，取最近的候选块"),
    "sparse": ("关键词检索", "BM25，对编号、金额、缩写更敏感"),
    "fusion": ("融合排序", "RRF 合并两路名次，不比较分数量纲"),
    "rerank": ("精排", "交叉编码器逐条重算相关性"),
    "retrieve": ("并行检索", "向量 + 关键词两路同时开跑，按名次融合"),
    "generate": ("生成回答", "只依据命中块作答并标注引用"),
}

_STATUS_COLOR = {
    "pending": ("#d8dee4", "#8b949e", "·"),
    "running": ("#1f6feb", "#1f6feb", "◐"),
    "done": ("#1a7f37", "#1a7f37", "✓"),
    "skip": ("#bf8700", "#bf8700", "–"),
    "error": ("#d1242f", "#d1242f", "!"),
}


def plan_steps(mode: str) -> list[dict]:
    """Pre-seed the whole plan as pending, so the panel is not empty while waiting."""
    keys = PLAN_STEPS.get(mode, PLAN_STEPS["fast"])
    return [{"key": key, "title": STEP_TITLES[key][0], "status": "pending", "detail": "", "meta": ""} for key in keys]


def render_trace(steps: list[dict], mode: str = "fast") -> str:
    """Vertical timeline of the pipeline. Every line carries a measured number."""
    if not steps:
        keys = PLAN_STEPS.get(mode, PLAN_STEPS["fast"])
        names = " → ".join(STEP_TITLES[key][0] for key in keys)
        note = (
            "提问后这里会逐条亮起，每条都带实测数字。"
            if not mode
            else f"提问后这里按顺序亮起：{names}。"
        )
        return (
            "<div style='color:#57606a;font-size:13px;padding:10px 4px'>" + note + "</div>"
        )

    done = sum(1 for step in steps if step.get("status") in ("done", "skip"))
    total = len(steps)
    rows = []
    for step in steps:
        status = step.get("status", "pending")
        color, text_color, mark = _STATUS_COLOR.get(status, _STATUS_COLOR["pending"])
        detail = step.get("detail") or ""
        meta = step.get("meta") or ""
        spin = "animation:rt-spin 1.1s linear infinite;display:inline-block;" if status == "running" else ""
        rows.append(
            "<div style='display:flex;gap:10px;padding:5px 0'>"
            f"<div style='flex:0 0 18px;height:18px;line-height:18px;text-align:center;"
            f"border-radius:9px;border:1px solid {color};color:{text_color};font-size:11px;"
            f"background:#fff;{spin}'>{mark}</div>"
            "<div style='flex:1 1 auto;min-width:0'>"
            f"<div style='font-size:13px;color:#1f2328;font-weight:600'>"
            f"{html.escape(step.get('title', ''))}"
            + (f"<span style='font-weight:400;color:#8b949e;margin-left:8px'>{html.escape(meta)}</span>" if meta else "")
            + "</div>"
            + (
                f"<div style='font-size:12px;color:#57606a;line-height:1.5;word-break:break-word'>"
                f"{html.escape(detail)}</div>"
                if detail
                else ""
            )
            + "</div></div>"
        )

    pct = done / total * 100 if total else 0
    return (
        "<style>@keyframes rt-spin{from{transform:rotate(0)}to{transform:rotate(360deg)}}</style>"
        "<div style=\"font-family:ui-sans-serif,system-ui,'Segoe UI',sans-serif;"
        "background:#f6f8fa;border:1px solid #d8dee4;border-radius:8px;padding:10px 12px\">"
        "<div style='display:flex;justify-content:space-between;align-items:center;"
        "font-size:11px;letter-spacing:.08em;color:#8b949e;margin-bottom:6px'>"
        f"<span>PIPELINE</span><span>{done}/{total}</span></div>"
        "<div style='height:3px;background:#e4e8ec;border-radius:2px;margin-bottom:1px'>"
        f"<div style='height:3px;width:{pct:.0f}%;background:#1a7f37;border-radius:2px'></div></div>"
        + "".join(rows)
        + "</div>"
    )


def _source_rows(sources: list[dict]) -> list[list]:
    rows = []
    for item in sources:
        scores = item.get("scores", {})

        def fmt(key: str) -> str:
            value = scores.get(key)
            return "—" if value is None else (f"{value:.3f}" if value < 1 else f"{value:.2f}")

        rows.append(
            [
                item["n"],
                item["filename"],
                item["page"] if item.get("page_to", item["page"]) == item["page"]
                else f"{item['page']}-{item['page_to']}",
                {"table": "表格", "figure": "图注", "ocr": "OCR"}.get(item["kind"], "正文"),
                fmt("fused"),
                fmt("dense"),
                fmt("sparse"),
                fmt("rerank"),
                " ".join(item["text"].split())[:110],
            ]
        )
    return rows


def _usage_line(result: dict) -> str:
    usage = result.get("usage") or {}
    if not usage.get("total_tokens"):
        return ""
    parts = [
        f"入 {usage.get('prompt_tokens', 0):,}",
        f"出 {usage.get('completion_tokens', 0):,}",
        f"合计 **{usage.get('total_tokens', 0):,}** tokens",
    ]
    if usage.get("tokens_per_second"):
        parts.append(f"{usage['tokens_per_second']:.1f} tok/s")
    line = "　｜　".join(parts)
    session = usage.get("session") or {}
    if session.get("total_tokens"):
        line += f"　｜　本次会话累计 {session['total_tokens']:,}"
    if session.get("priced") and session.get("cost"):
        line += f"　（≈ ¥{session['cost']:.4f}）"
    return f"\n- 用量：{line}"


def _result_markdown(result: dict) -> str:
    timings = result.get("timings", {})
    mode = (result.get("mode") or {}).get("label", "")
    lines = [
        f"- 模式：{mode}　｜　候选 {result.get('candidates', 0)} 块 → 采用 "
        f"{len(result.get('sources', []))} 块",
        "- 耗时："
        + "　".join(
            f"{label} {int(timings.get(key, 0))}ms"
            for key, label in (
                ("rewrite_ms", "改写"),
                ("embed_ms", "向量化"),
                ("retrieve_ms", "检索"),
                ("rerank_ms", "精排"),
                ("ttft_ms", "首字"),
                ("generate_ms", "生成"),
                ("total_ms", "合计"),
            )
            if timings.get(key)
        ),
    ]
    citations = result.get("citations") or {}
    if citations.get("total"):
        lines.append(f"- 引用：{citations['valid']}/{citations['total']} 个编号有效")
    if result.get("abstained"):
        lines.append("- 判定为资料不足（拒答）")
    if result.get("rerank", {}).get("used"):
        lines.append(f"- 精排：`{result['rerank'].get('model')}` 生效")
    elif result.get("rerank", {}).get("error"):
        lines.append(f"- 精排未生效：{result['rerank']['error']}")
    if result.get("degraded"):
        lines.append(f"- 降级：{result['degraded']}")
    return "\n".join(lines) + _usage_line(result)


def compare_markdown(report: dict) -> str:
    """Two modes, same question, retrieval only -- then one sentence of advice."""
    rows = report.get("rows", [])
    if not rows:
        return "没有可对比的结果。"
    fast, enhanced = rows[0], rows[1]
    lines = [
        f"**「{report['query']}」在两种模式下的检索结果**",
        "",
        "| 模式 | 命中块 | 覆盖页 | 页覆盖率 | 首位优势 | 精排 | 耗时 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['label']} | {row['hits']} | {row['pages']} | {row['coverage']:.0%} | "
            f"{row['margin']:.4f} | {'是' if row['rerank_used'] else '否'} | {row['ms']:.0f} ms |"
        )
    delta = enhanced["coverage"] - fast["coverage"]
    lines += [
        "",
        f"- 页覆盖率差 **{delta:+.0%}**（覆盖页 / 范围内全部页，越高说明证据面越广）",
        f"- 首位优势是第 1 名与其余命中块的平均分差，越大说明排序越有把握",
        "",
        f"> {report['advice']}",
    ]
    return "\n".join(lines)


def eval_markdown(result: dict) -> str:
    """Golden-set numbers for one mode, phrased as what they actually measure."""
    summary = result.get("summary") or {}
    rows = result.get("rows") or []
    if not rows:
        return "没有可评测的问题：请先入库示例文档，或检查 `eval/golden.jsonl` 的 doc 字段。"
    answerable = [row for row in rows if row.get("answerable", True)]
    abstain = len(rows) - len(answerable)
    missed = [row for row in answerable if not row["hit"]]
    tail = (
        f"- 有标注页码的题 {len(answerable)} 条，未进前 {summary['k']} 的 {len(missed)} 条"
        if missed
        else f"- 有标注页码的题 {len(answerable)} 条全部命中前 {summary['k']}"
    )
    if abstain:
        tail += f"；另有 {abstain} 条故意无答案，只检验拒答，不计入命中率"
    return (
        f"命中率 hit@{summary['k']} **{summary['hit']:.1%}**　｜　"
        f"首位命中 MRR **{summary['mrr']:.3f}**　｜　题目 {summary['questions']} 条\n\n"
        f"{tail}　｜　平均 {sum(row['ms'] for row in rows) / len(rows):.0f} ms/题"
    )


def _system_markdown(service: Service) -> str:
    info = service.system_info()
    usage = info["usage"]
    index = info["index"]
    price = (
        f"约 ¥{usage['cost']:.4f}（按配置单价估算）" if usage["priced"] else "未配置单价，仅统计 token"
    )
    ocr = info["ocr"]
    return (
        "### 运行状态\n\n"
        "| 项 | 值 |\n| --- | --- |\n"
        f"| 版本 / 启动 | `{info['version']}` / {info['started_at']} |\n"
        f"| 对话模型 | `{info['llm']['model']}`（{'已配置' if info['llm']['configured'] else '**未配置**'}） |\n"
        f"| 接口地址 | `{info['llm']['base_url']}` |\n"
        f"| 向量模型 | `{info['embedding']['model']}`（{'可用' if info['embedding']['configured'] else '未配置'}，"
        f"batch {info['embedding']['batch']} × 并发 {info['embedding']['concurrency']}） |\n"
        f"| 精排模型 | `{info['rerank']['model']}`（{'可用' if info['rerank']['configured'] else '未启用'}） |\n"
        f"| 视觉 · 扫描页 | `{ocr['model']}`（{ocr['state']}） |\n"
        f"| 视觉 · 图表 | `{ocr['vision_model']}`（并发 {ocr['concurrency']}） |\n\n"
        "### 索引与用量\n\n"
        "| 项 | 值 |\n| --- | --- |\n"
        f"| 文档 / 块 / 向量 | {index['docs']} / {index['chunks']} / {index['vectors']} |\n"
        f"| 向量维度 / 词表 | {index['dim']} / {index['vocab']} |\n"
        f"| 提问次数 | {usage['requests']} |\n"
        f"| tokens（入 / 出 / 合计） | {usage['llm_in']:,} / {usage['llm_out']:,} / {usage['total_tokens']:,} |\n"
        f"| 向量化 tokens | {usage['embed']:,}（{price}） |\n"
        f"| 接口访问保护 | {'需要 X-API-Key' if info['security']['api_key_required'] else '未开启'}"
        f"（与模型 Key 无关） |\n"
        f"| 界面登录 | {'需要密码' if info['security']['ui_login'] else '未开启'} |\n"
    )


def build_ui(service: Service) -> gr.Blocks:
    ready = settings.llm_configured
    vision_ok, vision_note = settings.vision_state
    header = (
        f"# 文档助手 v{VERSION}\n"
        f"对话 `{settings.llm_model}`（{'就绪' if ready else '**缺少 API Key**'}）"
        f"　｜　向量 `{settings.embed_model}`"
        f"　｜　视觉 `{settings.ocr_model}`（{'就绪' if vision_ok else vision_note}）"
        f"　｜　已索引 {service.library.stats()['docs']} 篇 / "
        f"{service.library.stats()['chunks']} 块"
    )

    with gr.Blocks(title=f"文档助手 v{VERSION}") as demo:
        gr.Markdown(header)

        doc_state = gr.State(None)
        page_state = gr.State([])
        chunk_state = gr.State([])
        source_state = gr.State([])
        history_state = gr.State([])

        # ---------------- 文档工作台 ----------------
        with gr.Tab("① 文档工作台"):
            flow = gr.HTML(flow_html(step_states(None, [])))

            with gr.Tabs():
                # ---- 1. 上传建索引 ----
                with gr.Tab(STEP_LABELS["upload"]):
                    with gr.Row():
                        with gr.Column(scale=5):
                            upload = gr.File(
                                label="上传文件（可多选）",
                                file_types=list(SUPPORTED_EXT),
                                file_count="multiple",
                            )
                            auto_index = gr.Checkbox(label="入库后自动建索引", value=True)
                            with gr.Accordion("顺便做增强识别（扫描页 OCR / 图表描述，会计费）", open=False):
                                do_enhance = gr.Checkbox(label="入库后立即增强", value=False)
                                limit_pages = gr.Slider(
                                    label="最多调用模型的页数",
                                    minimum=1,
                                    maximum=60,
                                    value=settings.ocr_max_pages,
                                    step=1,
                                )
                            upload_btn = gr.Button("解析并入库", variant="primary")
                            status = gr.Markdown(
                                "支持 PDF、Word、PPT、Excel、CSV、Markdown、TXT、HTML。"
                                f"视觉模型当前：{settings.ocr_model}（{vision_note}）。"
                            )
                        with gr.Column(scale=7):
                            doc_table = gr.Dataframe(
                                headers=DOC_HEADERS,
                                value=_doc_rows(service),
                                interactive=False,
                                wrap=True,
                                label="文档列表",
                            )
                    with gr.Row():
                        doc_dd = gr.Dropdown(
                            label="当前文档", choices=_doc_choices(service), value=None, scale=6
                        )
                        chunk_size = gr.Slider(
                            label="块长上限（字）", minimum=200, maximum=1200,
                            value=settings.chunk_size, step=50, scale=4,
                        )
                        reindex_btn = gr.Button("重建索引", scale=2)
                        delete_btn = gr.Button("删除", variant="stop", scale=1)

                # ---- 2. 解析结果 ----
                with gr.Tab(STEP_LABELS["parse"]):
                    gr.Markdown(
                        "解析决定后面一切：这一页告诉你文件被认成了什么格式、每页有多少文字、"
                        "哪些页会被当成扫描件或表格。"
                    )
                    parse_summary = gr.Markdown("选择文档后显示。")
                    parse_strip = gr.HTML("")
                    parse_note = gr.Markdown("")

                # ---- 3. 分块地图 ----
                with gr.Tab(STEP_LABELS["chunks"]):
                    gr.Markdown(
                        "每个块都是一次检索的最小单位。条形长度＝字数，颜色＝块类型；"
                        "上方页轴显示全文字量分布，问答后同一张图会把被引用的块标红。"
                    )
                    chunk_summary = gr.Markdown()
                    chunk_map = gr.HTML(chunk_map_html([]))
                    with gr.Row():
                        chunk_dd = gr.Dropdown(label="查看块", choices=[], value=None, scale=3)
                        chunk_box = gr.Textbox(label="块全文", lines=8, max_lines=24, scale=9)

                # ---- 4. 页级诊断 ----
                with gr.Tab(STEP_LABELS["pages"]):
                    gr.Markdown(
                        "扫描页会走 OCR，表格页会走 pdfplumber 精查，图形页会走图注模型——"
                        "这里的标记就是路由依据。"
                    )
                    page_table = gr.Dataframe(
                        headers=PAGE_HEADERS, interactive=False, wrap=True,
                        label="每页的文本量与结构信号",
                    )

                # ---- 5. 原文查看 ----
                with gr.Tab(STEP_LABELS["text"]):
                    with gr.Row():
                        view_dd = gr.Dropdown(label="页码", choices=[], value=None, scale=2)
                        page_box = gr.Textbox(label="单页原文", lines=18, max_lines=40, scale=10)

                # ---- 6. 检索指标 ----
                with gr.Tab(STEP_LABELS["metrics"]):
                    gr.Markdown(
                        "用 `eval/golden.jsonl` 里带标注页码的题目，在**这份文档**上分别跑两种模式，"
                        "只做检索、不调用生成模型。命中率回答「找得到吗」，MRR 回答「排得靠前吗」。"
                    )
                    with gr.Row():
                        metrics_btn = gr.Button("跑一次两模式评测", variant="primary", scale=2)
                        metrics_scope = gr.Dropdown(
                            label="文档范围（留空 = 全部）",
                            choices=_doc_choices(service),
                            multiselect=True,
                            value=[],
                            scale=6,
                        )
                    metrics_md = gr.Markdown(
                        "还没有跑过。示例库上的参考值：快速 MRR 0.972、增强 MRR 1.000（21 题）。"
                    )

            with gr.Accordion("运行状态与用量", open=False):
                with gr.Row():
                    key_box = gr.Textbox(
                        label="临时模型 Key（只存内存，不落盘；重启失效）",
                        type="password",
                        placeholder="sk-...",
                        scale=8,
                    )
                    key_btn = gr.Button("应用", scale=1)
                    sys_btn = gr.Button("刷新状态", scale=2)
                key_md = gr.Markdown(
                    "克隆仓库后没有 `.env` 也能用：在这里贴一次 Key 即可。"
                    "问答、向量、精排、视觉共用这一把，长期使用建议写进 `.env` 的 `SILICONFLOW_API_KEY`。"
                )
                sys_md = gr.Markdown(_system_markdown(service))

        # ---------------- 提问 ----------------
        with gr.Tab("② 提问"):
            with gr.Row():
                mode = gr.Radio(
                    label="检索方式",
                    choices=MODE_CHOICES,
                    value="fast",
                    scale=5,
                )
                chat_scope = gr.Dropdown(
                    label="范围（留空＝全部文档）",
                    choices=_doc_choices(service),
                    multiselect=True,
                    value=[],
                    scale=4,
                )
            gr.Markdown(
                "**快速检索**：两路检索并行、按名次融合后直接作答，约 1 秒出首字。\n\n"
                "**增强检索**：先补全追问，再把候选池放大一倍交给交叉编码器重排，"
                "多花一次改写和一次精排的时间。不确定时先用下面的「对比两种模式」。"
            )

            with gr.Row():
                with gr.Column(scale=6):
                    chatbot = gr.Chatbot(label="对话", height=420, show_label=True)
                    with gr.Row():
                        chat_input = gr.Textbox(
                            label="问题",
                            placeholder="例如：付款账期最长是多少天？",
                            scale=9,
                            container=True,
                        )
                        send_btn = gr.Button("发送", variant="primary", scale=1)
                        clear_btn = gr.Button("清空", scale=1)
                with gr.Column(scale=6):
                    trace = gr.HTML(render_trace(plan_steps("fast"), "fast"))

            answer_meta = gr.Markdown()
            with gr.Accordion("本次命中的块", open=True):
                hit_map = gr.HTML(chunk_map_html([]))
                source_table = gr.Dataframe(
                    headers=SOURCE_HEADERS, interactive=False, wrap=True, label="命中列表"
                )
                with gr.Row():
                    source_dd = gr.Dropdown(label="展开原文", choices=[], value=None, scale=2)
                    source_box = gr.Textbox(label="块原文", lines=8, max_lines=24, scale=10)

            with gr.Accordion("要不要用增强检索？", open=False):
                compare_btn = gr.Button("对比两种模式给这个问题", scale=2)
                compare_md = gr.Markdown(
                    "上一次提问之后点这里：两种模式各检索一次（不调用生成模型），"
                    "直接看增强多带回了哪些页、又挤掉了哪些页。"
                )

        # ================= handlers =================

        def on_upload(files, also_index, also_enhance, page_limit, current, progress=gr.Progress()):
            paths = [p for p in _file_paths(files) if Path(p).suffix.lower() in SUPPORTED_EXT]
            doc = service.docs.get(current) if current else None
            chunks = [c for c in service.library.records if doc and c.doc_id == doc["id"]]
            if not paths:
                return (
                    "请先选择文件。", _doc_rows(service), _update(choices=_doc_choices(service)),
                    doc, chunks, chunk_map_html(chunks), chunk_summary_md(chunks),
                    _update(choices=[(f"#{c.idx} · {c.chars} 字 · {c.kind}", c.idx) for c in chunks]),
                    _update(choices=_doc_choices(service), value=[]),
                    flow_html(step_states(doc, chunks)),
                )

            progress(0.05, desc=f"解析 {len(paths)} 个文件")
            results = service.ingest_files(paths)
            lines: list[str] = ["### 入库结果", ""]
            last_id = None

            for result in results:
                if not result.get("ok"):
                    lines.append(f"- **{result['filename']}** 解析失败：{result['error']}")
                    continue
                doc = result["doc"]
                last_id = doc["id"]
                lines.append(
                    f"- {'已存在' if result['existed'] else '已入库'} **{doc['filename']}**"
                    f"（{doc.get('format', '')}）：{doc['page_count']} 页 / "
                    f"{doc['chars']:,} 字符，用时 {result['ms']} ms"
                )
                if also_index:
                    try:
                        info = service.index(doc["id"], force=True)
                        note = (
                            f"  - 索引 {info['chunks']} 块 / {info['dim']} 维"
                            f"（{info['strategy']}，向量化 {info['embed_ms']:.0f} ms）"
                        )
                        if info["embed_error"]:
                            note += f"\n  - 向量化失败，已退化为纯关键词索引：{info['embed_error']}"
                        lines.append(note)
                    except ServiceError as exc:
                        lines.append(f"  - 建索引失败：{exc}")

                if also_enhance:
                    if not settings.ocr_ready:
                        lines.append(f"  - 跳过增强：视觉模型{settings.vision_state[1]}")
                    else:
                        try:
                            report = service.enhance(doc["id"], limit_pages=int(page_limit))
                            lines.append(
                                f"  - 增强：调用 {settings.ocr_model} {report['api_calls']} 次，"
                                f"{report['text_before']:,} → {report['text_after']:,} 字符，"
                                f"{report['ms'] / 1000:.1f}s（并发 {report['concurrency']}）"
                            )
                            if report["api_calls"]:
                                service.index(doc["id"], force=True)
                        except Exception as exc:
                            lines.append(f"  - 增强失败：{exc}")

            progress(1.0, desc="完成")
            doc = service.docs.get(last_id) or {}
            chunks = [c for c in service.library.records if last_id and c.doc_id == last_id]
            return (
                "\n".join(lines),
                _doc_rows(service),
                _update(choices=_doc_choices(service), value=last_id),
                last_id,
                chunks,
                chunk_map_html(chunks),
                chunk_summary_md(chunks) if chunks else "尚未建立索引。",
                _update(
                    choices=[(f"#{c.idx} · {c.chars} 字 · {c.kind}", c.idx) for c in chunks],
                    value=None,
                ),
                _update(choices=_doc_choices(service), value=[]),
                flow_html(step_states(doc, chunks)),
            )

        def on_doc_change(doc_id):
            doc = service.docs.get(doc_id) if doc_id else None
            if doc is None:
                return (
                    "请选择或上传一份文件。", "", "", [], [], chunk_map_html([]), "",
                    _update(choices=[]), _update(choices=[]), "",
                    flow_html(step_states(None, [])),
                )
            rows, pages = _page_rows(doc)
            chunks = [c for c in service.library.records if c.doc_id == doc["id"]]
            if doc["scan_like_pages"]:
                note = (
                    f"- 有 {len(doc['scan_like_pages'])} 页文字量低于阈值，"
                    "用视觉模型重读会更完整：勾选「入库后立即增强」，或换一份更清晰的扫描件。"
                )
            else:
                note = "- 文本层完整，没有需要 OCR 的页；表格页与图形页的信号见第 4 步。"
            return (
                f"当前文档：**{doc['filename']}**（{doc['page_count']} 页 · "
                f"{len(chunks)} 块）。解析细节见「② 解析结果」。",
                _doc_summary(doc),
                page_strip_html(doc),
                rows,
                pages,
                chunk_map_html(chunks),
                chunk_summary_md(chunks) or "尚未建立索引：回到第 1 步点「重建索引」。",
                _update(choices=[(f"#{c.idx} · {c.chars} 字 · {c.kind}", c.idx) for c in chunks], value=None),
                _update(choices=pages, value=None),
                note,
                flow_html(step_states(doc, chunks)),
            )

        def on_delete(doc_id):
            choices = _doc_choices(service)
            if not doc_id:
                return (
                    "请先选择文档。", _doc_rows(service), _update(choices=choices), None,
                    [], chunk_map_html([]), "", _update(choices=[]), _update(choices=[]), "",
                    flow_html(step_states(None, [])), "", "", [], _update(choices=choices, value=[]),
                )
            name = service.docs.get(doc_id, {}).get("filename", doc_id)
            service.delete(doc_id)
            choices = _doc_choices(service)
            return (
                f"已删除 **{name}** 及其索引。",
                _doc_rows(service),
                _update(choices=choices, value=None),
                None,
                [],
                chunk_map_html([]),
                "",
                _update(choices=[], value=None),
                _update(choices=[], value=None),
                "",
                flow_html(step_states(None, [])),
                "",
                "",
                [],
                _update(choices=choices, value=[]),
            )

        def on_reindex(doc_id, size):
            if not doc_id:
                return "请先选择文档。", _doc_rows(service), [], chunk_map_html([]), "", _update(choices=[]), flow_html(step_states(None, []))
            try:
                info = service.index(doc_id, size=int(size), force=True)
            except ServiceError as exc:
                doc = service.docs.get(doc_id)
                return f"建索引失败：{exc}", _doc_rows(service), [], chunk_map_html([]), "", _update(choices=[]), flow_html(step_states(doc, []))
            chunks = [c for c in service.library.records if c.doc_id == doc_id]
            note = (
                f"已重建索引：{info['chunks']} 块 / {info['dim']} 维"
                f"（{info['strategy']}，向量化 {info['embed_ms']:.0f} ms，"
                f"总耗时 {info['ms']:.0f} ms）"
            )
            if info["embed_error"]:
                note += f"\n\n> 向量化未生效：{info['embed_error']}\n> 当前只能用关键词检索。"
            doc = service.docs.get(doc_id)
            return (
                note,
                _doc_rows(service),
                chunks,
                chunk_map_html(chunks),
                chunk_summary_md(chunks),
                _update(choices=[(f"#{c.idx} · {c.chars} 字 · {c.kind}", c.idx) for c in chunks], value=None),
                flow_html(step_states(doc, chunks)),
            )

        def on_chunk_view(chunk_idx, chunks):
            if not chunks:
                return "请先建立索引。"
            if chunk_idx is None:
                return "请选择一个块。"
            for chunk in chunks:
                if chunk.idx == int(chunk_idx):
                    return (
                        f"[#{chunk.idx} · 第 {chunk.page_from}-{chunk.page_to} 页 · "
                        f"{chunk.kind} · {chunk.chars} 字]\n章节：{chunk.section or '—'}\n\n{chunk.text}"
                    )
            return "未找到该块。"

        def on_page_view(page_no, doc_id):
            doc = service.docs.get(doc_id) if doc_id else None
            if doc is None:
                return "请先选择文档。"
            if not page_no:
                return "请选择页码。"
            return page_text_view(doc, int(page_no))

        def _source_choices(sources: list[dict]) -> list[tuple[str, int]]:
            return [
                (
                    f"[{item['n']}] 《{item['filename']}》p.{item['page']} · {item['chars']} 字",
                    item["n"],
                )
                for item in sources
            ]

        def on_mode_change(mode_value):
            return render_trace(plan_steps(mode_value), mode_value)

        def on_chat(message, history, mode_value, scope):
            history = list(history or [])
            pool = [
                record
                for record in service.library.records
                if not scope or record.doc_id in scope
            ]
            plan = plan_steps(mode_value)
            blank = (
                history, "", render_trace(plan, mode_value), "", [], chunk_map_html(pool),
                history, [], _update(choices=[], value=None),
            )
            if not message or not message.strip():
                yield blank
                return
            if not service.library.records:
                history = history + [
                    {"role": "user", "content": message},
                    {"role": "assistant", "content": "索引为空：请先在「文档工作台」上传文件并建立索引。"},
                ]
                yield (
                    history, "", render_trace(plan, mode_value), "索引为空，无法检索。", [],
                    chunk_map_html([]), history, [], _update(choices=[], value=None),
                )
                return

            past = history[-settings.history_turns * 2 :]
            history = history + [{"role": "user", "content": message}]
            steps = {item["key"]: item for item in plan}
            order = [item["key"] for item in plan]
            sources: list[dict] = []
            answer = ""
            meta_md = ""
            rails: dict[str, str] = {}

            def step(key: str, status: str = "running", detail: str = "", meta: str = "") -> None:
                if key not in steps:
                    steps[key] = {"key": key, "title": STEP_TITLES[key][0]}
                    order.append(key)
                steps[key].update(status=status, detail=detail, meta=meta)

            def view(show_map: bool = True):
                return (
                    history + [{"role": "assistant", "content": answer or "…"}],
                    "",
                    render_trace([steps[k] for k in order], mode_value),
                    meta_md,
                    _source_rows(sources),
                    chunk_map_html(pool, sources) if (show_map and sources) else chunk_map_html([]),
                    history,
                    sources,
                    _update(choices=_source_choices(sources), value=None),
                )

            try:
                for event in service.ask_stream(
                    message,
                    doc_ids=scope or None,
                    history=past,
                    mode=mode_value,
                ):
                    stage = event["stage"]

                    if stage == "rewrite_start":
                        step("rewrite", "running", "读取对话历史，判断是否需要补全指代…")
                        yield view(False)

                    elif stage == "rewrite_done":
                        if event.get("rewritten"):
                            step("rewrite", "done", f"改写为：{event['query']}", f"{event['ms']:.0f} ms")
                        elif event.get("skipped"):
                            step("rewrite", "skip", "未配置 Key，跳过改写", f"{event['ms']:.0f} ms")
                        elif event.get("error"):
                            step("rewrite", "error", event["error"], f"{event['ms']:.0f} ms")
                        else:
                            step("rewrite", "skip", "问题已完整，不需要改写", f"{event['ms']:.0f} ms")
                        yield view(False)

                    elif stage == "embed_start":
                        step("embed", "running", f"模型 {event.get('model')}")
                        yield view(False)

                    elif stage == "embed_done":
                        if event.get("error"):
                            step("embed", "error", event["error"], f"{event['ms']:.0f} ms")
                        elif event.get("skipped"):
                            step("embed", "skip", "未配置向量模型，只用关键词检索", "0 ms")
                        else:
                            step(
                                "embed",
                                "done",
                                f"{event['dim']} 维 · L2 范数 {event['norm']}（已归一化）"
                                f" · 消耗 {event['tokens']} tokens",
                                f"{event['ms']:.0f} ms",
                            )
                        yield view(False)

                    elif stage == "dense_start":
                        if mode_value == "fast":
                            step("retrieve", "running", f"与 {event.get('vectors', 0)} 个块向量比余弦…")
                        else:
                            step("dense", "running", f"与 {event.get('vectors', 0)} 个块向量比较余弦相似度")
                        yield view(False)

                    elif stage == "dense_done":
                        tops = "；".join(
                            f"p.{item['page']} {item['score']}" for item in event.get("top", [])
                        )
                        if mode_value == "fast":
                            rails["dense"] = f"向量 {event['hits']} 块，最高 {tops}"
                            step("retrieve", "running", rails["dense"])
                        else:
                            step("dense", "done", f"命中 {event['hits']} 块，最高相似度 {tops}", f"{event['ms']:.1f} ms")
                        yield view(False)

                    elif stage == "sparse_start":
                        terms = "、".join(event.get("terms") or []) or "（无有效词）"
                        if mode_value != "fast":
                            step("sparse", "running", f"分词：{terms}")
                        yield view(False)

                    elif stage == "sparse_done":
                        tops = "；".join(
                            f"p.{item['page']} {item['score']}" for item in event.get("top", [])
                        )
                        if mode_value == "fast":
                            rails["sparse"] = f"关键词 {event['hits']} 块，最高 {tops}"
                            step("retrieve", "running", "；".join(rails.values()))
                        else:
                            step("sparse", "done", f"命中 {event['hits']} 块，最高分 {tops}", f"{event['ms']:.1f} ms")
                        yield view(False)

                    elif stage == "fuse_done":
                        detail = (
                            f"向量 {event['dense']} 条 + 关键词 {event['sparse']} 条"
                            f" → {event['candidates']} 个候选（k={event['k']}）"
                        )
                        if mode_value == "fast":
                            step("retrieve", "done", detail, f"{event['ms']:.1f} ms")
                        else:
                            step("fusion", "done", detail, f"{event['ms']:.1f} ms")
                        yield view(False)

                    elif stage == "rerank_start":
                        step("rerank", "running", f"逐条重算 {event['candidates']} 个候选的相关性")
                        yield view(False)

                    elif stage == "rerank_done":
                        if event.get("error"):
                            step("rerank", "error", event["error"], f"{event['ms']:.0f} ms")
                        else:
                            tops = "；".join(
                                f"p.{item['page']} {item['score']}" for item in event.get("top", [])
                            )
                            step("rerank", "done", f"精排最高分 {tops}", f"{event['ms']:.0f} ms")
                        yield view(False)

                    elif stage == "rerank_skipped":
                        step("rerank", "skip", event.get("reason", ""), "0 ms")
                        yield view(False)

                    elif stage == "retrieved":
                        sources = event["sources"]
                        meta_md = (
                            f"候选 {event['candidates']} 个块，采用 {len(sources)} 个"
                            + ("（同页重复已去重）" if event["candidates"] > len(sources) else "")
                        )
                        yield view()

                    elif stage == "delta":
                        first = not answer
                        answer += event["text"]
                        if first:
                            step(
                                "generate",
                                "running",
                                "首个字已到达，正在流式输出…",
                                f"首字 {event.get('ttft_ms', 0):.0f} ms",
                            )
                        yield view()

                    elif stage == "error":
                        step("generate", "error", event["error"], "")
                        history = history + [
                            {"role": "assistant", "content": f"调用模型失败：{event['error']}"}
                        ]
                        yield (
                            history, "", render_trace([steps[k] for k in order], mode_value),
                            meta_md, _source_rows(sources), chunk_map_html(pool, sources), history,
                            sources, _update(choices=_source_choices(sources), value=None),
                        )
                        return

                    elif stage == "done":
                        result = event["result"]
                        sources = result["sources"]
                        history = history + [{"role": "assistant", "content": result["answer"]}]
                        timings = result["timings"]
                        step(
                            "generate",
                            "done",
                            f"{result['usage'].get('completion_tokens', 0)} tokens，"
                            f"首字 {timings.get('ttft_ms', 0):.0f} ms，"
                            f"总耗时 {timings.get('total_ms', 0):.0f} ms",
                            "完成",
                        )
                        meta_md = _result_markdown(result)
                        yield (
                            history,
                            "",
                            render_trace([steps[k] for k in order], mode_value),
                            meta_md,
                            _source_rows(sources),
                            chunk_map_html(pool, sources),
                            history,
                            sources,
                            _update(choices=_source_choices(sources), value=None),
                        )
                        return
            except Exception as exc:  # a pipeline crash must not leave a spinner
                log.exception("chat failed")
                step("generate", "error", str(exc), "")
                history = history + [{"role": "assistant", "content": f"处理失败：{exc}"}]
                yield (
                    history, "", render_trace([steps[k] for k in order], mode_value), meta_md,
                    _source_rows(sources), chunk_map_html(pool, sources), history,
                    sources, _update(choices=_source_choices(sources), value=None),
                )

        def on_clear(mode_value):
            return (
                [], "", render_trace(plan_steps(mode_value), mode_value), "", [], chunk_map_html([]),
                [], [], _update(choices=[], value=None),
            )

        def on_source_view(number, sources):
            if not sources:
                return "本次没有引用来源。"
            if number is None:
                return "请选择一条引用。"
            for item in sources:
                if item["n"] == int(number):
                    scores = item["scores"]
                    vector = "—" if scores["dense"] is None else f"{scores['dense']:.3f}"
                    rerank = "—" if scores["rerank"] is None else f"{scores['rerank']:.3f}"
                    return (
                        f"[{item['n']}] 《{item['filename']}》p.{item['page']} · {item['kind']}"
                        f" · {item['chars']} 字\n"
                        f"融合 {scores['fused']:.4f}　向量 {vector}　精排 {rerank}\n"
                        f"章节：{item.get('section') or '—'}\n\n{item['text']}"
                    )
            return "未找到该引用。"

        def on_compare(history, scope):
            history = list(history or [])
            question = ""
            for item in reversed(history):
                if item.get("role") == "user":
                    question = item.get("content") or ""
                    break
            if not question.strip():
                return "先在上面提一个问题，再回来对比两种模式。"
            try:
                report = service.compare_modes(question, doc_ids=scope or None, top_k=5)
            except ServiceError as exc:
                return f"对比失败：{exc}"
            return compare_markdown(report)

        def on_metrics(scope, progress=gr.Progress()):
            doc_ids = scope or None
            labels = {"fast": MODE_CHOICES[0][0], "enhanced": MODE_CHOICES[1][0]}
            parts: list[str] = []
            for index, mode_name in enumerate(("fast", "enhanced"), start=1):
                progress(index / 3, desc=f"评测：{labels[mode_name]}")
                try:
                    result = service.evaluate(top_k=5, mode=mode_name, doc_ids=doc_ids)
                except Exception as exc:
                    return f"评测失败：{exc}"
                parts.append(f"**{labels[mode_name]}**　{eval_markdown(result)}")
            progress(1.0, desc="完成")
            return (
                "\n\n".join(parts)
                + "\n\n> 两组数字用同一批题目、同一个 top_k，只有候选池宽度和精排开关不同，"
                "所以差值可以直接读成「增强换来了什么」。"
            )

        # ================= bindings =================

        upload_btn.click(
            fn=on_upload,
            inputs=[upload, auto_index, do_enhance, limit_pages, doc_state],
            outputs=[
                status, doc_table, doc_dd, doc_state, chunk_state, chunk_map,
                chunk_summary, chunk_dd, chat_scope, flow,
            ],
            show_progress=True,
        )
        doc_dd.change(
            fn=on_doc_change,
            inputs=[doc_dd],
            outputs=[
                status, parse_summary, parse_strip, page_table, page_state, chunk_map,
                chunk_summary, chunk_dd, view_dd, parse_note, flow,
            ],
            show_progress=True,
        )
        view_dd.change(fn=on_page_view, inputs=[view_dd, doc_state], outputs=page_box)
        delete_btn.click(
            fn=on_delete,
            inputs=[doc_state],
            outputs=[
                status, doc_table, doc_dd, doc_state, chunk_state, chunk_map, chunk_summary,
                chunk_dd, view_dd, parse_note, flow, parse_summary, parse_strip,
                page_table, chat_scope,
            ],
        )
        reindex_btn.click(
            fn=on_reindex,
            inputs=[doc_state, chunk_size],
            outputs=[status, doc_table, chunk_state, chunk_map, chunk_summary, chunk_dd, flow],
            show_progress=True,
        )
        chunk_dd.change(fn=on_chunk_view, inputs=[chunk_dd, chunk_state], outputs=chunk_box)
        mode.change(fn=on_mode_change, inputs=[mode], outputs=trace)
        metrics_btn.click(
            fn=on_metrics, inputs=[metrics_scope], outputs=metrics_md, show_progress=True
        )

        _chat_outputs = [
            chatbot, chat_input, trace, answer_meta,
            source_table, hit_map, history_state, source_state, source_dd,
        ]
        send_btn.click(
            fn=on_chat,
            inputs=[chat_input, history_state, mode, chat_scope],
            outputs=_chat_outputs,
            show_progress="hidden",
        )
        chat_input.submit(
            fn=on_chat,
            inputs=[chat_input, history_state, mode, chat_scope],
            outputs=_chat_outputs,
            show_progress="hidden",
        )
        clear_btn.click(fn=on_clear, inputs=[mode], outputs=_chat_outputs)
        source_dd.change(fn=on_source_view, inputs=[source_dd, source_state], outputs=source_box)
        compare_btn.click(fn=on_compare, inputs=[history_state, chat_scope], outputs=compare_md)
        sys_btn.click(fn=lambda: _system_markdown(service), inputs=None, outputs=sys_md)

        def on_key(key):
            value = (key or "").strip()
            if not value:
                return "未填写 Key。", _system_markdown(service)
            settings.llm_api_key = value
            settings.ocr_enabled = True
            if not settings.rerank_enabled:
                settings.rerank_enabled = True
            return (
                f"已应用 `{value[:6]}…{value[-4:]}`：问答、向量、精排、视觉都改用它。"
                "重启后失效，若要长期生效请写进 `.env`。",
                _system_markdown(service),
            )

        key_btn.click(fn=on_key, inputs=[key_box], outputs=[key_md, sys_md])

    return demo
