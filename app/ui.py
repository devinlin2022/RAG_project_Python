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

from app.style import CSS, HEAD
from app.style import _ICON_PATHS as ICON_PATHS
from rag.config import VERSION, settings
from rag.diagnose import reset_cache as reset_diagnosis
from rag.embed.client import breaker_state, reset_breaker
from rag.ingest.extract import SUPPORTED_EXT
from rag.ingest.loader import page_text as page_text_view
from rag.service import Service, ServiceError
from rag.utils import format_pages

log = logging.getLogger(__name__)

DOC_HEADERS = ["文件名", "格式", "页", "块", "状态"]
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
    ("upload", "上传建索引", "文件解析、切块、算向量"),
    ("parse", "解析结果", "格式识别与逐页文本量"),
    ("chunks", "分块地图", "每块的位置、类型、字数"),
    ("pages", "页级诊断", "每页的扫描 / 表格 / 图形信号"),
    ("text", "原文查看", "逐页核对识别结果"),
    ("metrics", "检索指标", "两种模式在这份文档上的实测表现"),
]
STEP_LABELS = {key: label for key, label, _ in STEPS}

KIND_STYLE = {
    "text": ("正文", "#1a56db"),
    "table": ("表格", "#057a55"),
    "figure": ("图注", "#b45309"),
    "ocr": ("OCR", "#7e3af2"),
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
            # 业务视角只看"这份文档进来了没有、切了几块"。
            # 内部 ID、字符数、是否增强放到「解析明细」里，别挤在这张表上。
            row["filename"],
            row["format"],
            row["pages"],
            row["chunks"],
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


def _bar(pct: float, color: str, height: int = 7) -> str:
    """One proportional bar inside a shared track."""
    return (
        f"<span class='fb-track'><i style='width:{max(pct, 1.5):.2f}%;"
        f"background:{color}'></i></span>"
    )


def flow_html(steps: list[dict]) -> str:
    """The six-step directory as a stepper: what has been done to this file.

    A tab bar tells you what exists; this tells you what has actually been done
    to the document, which is the part a reader cannot infer from a filename.
    """
    chips = []
    for index, step in enumerate(steps):
        done = step.get("done")
        chips.append(
            f"<div class='fb-step-chip{' done' if done else ''}'>"
            f"<b>{'✓' if done else index + 1}　{html.escape(step['label'])}</b>"
            f"<span>{html.escape(step.get('hint', ''))}</span></div>"
        )
    return "<div class='fb-stepper'>" + "".join(chips) + "</div>"


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
        tone = f"rgba(26,86,219,{0.12 + 0.6 * shade:.2f})" if page["chars"] else ""
        flag = page.get("scan_like") and "扫描" or page.get("table_like") and "表格" or ""
        title = f"第 {page['page']} 页 · {page['chars']} 字符" + (f" · {flag}" if flag else "")
        cells.append(
            f"<i title='{html.escape(title)}'" + (f" style='background:{tone}'" if tone else "") + "></i>"
        )
    thin = sum(1 for value in chars if value < settings.scan_page_char_threshold)
    return (
        "<div class='fb-strip'>" + "".join(cells) + "</div>"
        "<div class='fb-strip-note'>"
        f"<span>{len(chars)} 页</span><span>颜色越深文字越多</span>"
        f"<span>低于 {settings.scan_page_char_threshold} 字的页 {thin} 页</span></div>"
    )


def chunk_map_html(
    chunks: list, hits: list[dict] | None = None, limit: int = 240, title: str = "分块地图"
) -> str:
    """Blocks as proportional bars: what the splitter produced, and what answered."""
    if not chunks:
        return (
            "<div class='fb-card'><div class='fb-card-head'>"
            f"<div><div class='fb-card-title'>{title}</div></div></div>"
            "<div class='fb-empty'>还没有分块。上传文件并建立索引后，"
            "这里会显示每个块的位置与大小。</div></div>"
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
        tone = f"rgba(26,86,219,{0.12 + 0.6 * shade:.2f})" if shade else ""
        cited = bool(hits) and any(
            item["page"] <= page <= item.get("page_to", item["page"]) for item in hits
        )
        cells.append(
            f"<i class='{'cited' if cited else ''}' title='第 {page} 页 · {per_page.get(page, 0)} 字符'"
            + (f" style='background:{tone}'" if tone else "")
            + "></i>"
        )

    rows = []
    for chunk in chunks[:limit]:
        hit = rank.get(f"{chunk.doc_id}:{chunk.page_from}")
        label, color = KIND_STYLE.get(chunk.kind, KIND_STYLE["text"])
        width = chunk.chars / biggest * 100
        badge = f"<span class='fb-cite' title='引用 {hit}'>{hit}</span>" if hit else ""
        span = f"p.{chunk.page_from}"
        if chunk.page_to != chunk.page_from:
            span += f"-{chunk.page_to}"
        preview = html.escape(" ".join(chunk.text.split())[:84])
        rows.append(
            f"<div class='fb-map-row{' cited' if hit else ''}'>"
            f"<span class='fb-map-idx'>#{chunk.idx}</span>"
            f"<span class='fb-map-page'>{span}</span>"
            f"<span class='fb-map-kind'>{label}</span>"
            f"{_bar(width, color)}"
            f"<span class='fb-map-chars'>{chunk.chars} 字</span>"
            f"<div class='fb-map-preview'>{preview}{badge}</div>"
            "</div>"
        )

    more = (
        f"<div class='fb-strip-note'>共 {len(chunks)} 块，仅显示前 {limit} 块</div>"
        if len(chunks) > limit
        else ""
    )

    legend = "".join(
        f"<span><b style='background:{color}'></b>{name}</span>"
        for name, color in KIND_STYLE.values()
    )

    sub = "按字数成比例 · 红框为被引用页" if hits else "按字数成比例 · 颜色为块类型"
    return (
        "<div class='fb-card'><div class='fb-card-head'>"
        f"<div><div class='fb-card-title'>{title}</div>"
        f"<div class='fb-card-sub'>{sub}</div></div>"
        f"<span class='fb-badge'>{len(chunks)} 块</span></div>"
        "<div class='fb-card-body tight'>"
        "<div class='fb-strip'>" + "".join(cells) + "</div>"
        "<div class='fb-strip-note'>"
        f"<span>{pages} 页</span><span>颜色越深文字越多</span>"
        + ("<span>红框＝被引用页</span>" if hits else "")
        + "</div>"
        "<div class='fb-map'>" + "".join(rows) + more + "</div>"
        "<div class='fb-legend'>" + legend + "</div>"
        "</div></div>"
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

_STATUS_MARK = {"pending": "·", "running": "<i></i>", "done": "✓", "skip": "–", "error": "!"}

_LEAN_FOOT = (
    "<div class='fb-trace-foot'>快速检索到此为止：两路检索并行、按名次融合后直接作答。"
    "切到<b>增强检索</b>会多出改写、BM25 分词、RRF 融合与精排前后分对比。</div>"
)


def plan_steps(mode: str) -> list[dict]:
    """Pre-seed the whole plan as pending, so the panel is not empty while waiting."""
    keys = PLAN_STEPS.get(mode, PLAN_STEPS["fast"])
    return [{"key": key, "title": STEP_TITLES[key][0], "status": "pending", "detail": "", "meta": ""} for key in keys]


def render_trace(steps: list[dict], mode: str = "fast") -> str:
    """Vertical timeline of the pipeline. Every line carries a measured number."""
    title = "增强检索流水线" if mode == "enhanced" else "快速检索"
    if not steps:
        keys = PLAN_STEPS.get(mode, PLAN_STEPS["fast"])
        names = " → ".join(STEP_TITLES[key][0] for key in keys)
        note = (
            "提问后这里会逐条亮起，每条都带实测数字。"
            if not mode
            else f"提问后这里按顺序亮起：{names}。"
        )
        return (
            "<div class='fb-card'><div class='fb-card-head'>"
            f"<div><div class='fb-card-title'>{title}</div>"
            f"<div class='fb-card-sub'>{len(keys)} 步 · 每题实时计时</div></div>"
            f"<span class='fb-badge'>{len(keys)} 步</span></div>"
            f"<div class='fb-card-body'><div class='fb-trace'><div class='fb-empty'>{note}</div>"
            "</div></div></div>"
        )

    done = sum(1 for step in steps if step.get("status") in ("done", "skip"))
    total = len(steps)
    rows = []
    for step in steps:
        status = step.get("status", "pending")
        detail = step.get("detail") or ""
        meta = step.get("meta") or ""
        rows.append(
            f"<div class='fb-step {html.escape(status)}'>"
            f"<div class='fb-step-dot'>{_STATUS_MARK.get(status, '·')}</div>"
            "<div class='fb-step-body'>"
            f"<div class='fb-step-name'>{html.escape(step.get('title', ''))}"
            + (f"<em>{html.escape(meta)}</em>" if meta else "")
            + "</div>"
            + (f"<div class='fb-step-detail'>{html.escape(detail)}</div>" if detail else "")
            + "</div></div>"
        )

    pct = done / total * 100 if total else 0
    foot = _LEAN_FOOT if mode == "fast" and all(
        step.get("status") in ("done", "skip", "error") for step in steps
    ) else ""
    return (
        "<div class='fb-card'><div class='fb-card-head'>"
        f"<div><div class='fb-card-title'>{title}</div>"
        f"<div class='fb-card-sub'>{total} 步 · 每一步都带实测数字</div></div>"
        f"<span class='fb-badge {'green' if done == total else ''}'>{done}/{total}</span></div>"
        "<div class='fb-card-body'>"
        "<div class='fb-trace'>"
        "<div class='fb-progress'><i style='width:%.0f%%'></i></div>" % pct
        + "".join(rows)
        + "</div>"
        + foot
        + "</div></div>"
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


# ------------------------------------------------------------------ shell


def icon(name: str, size: int = 16) -> str:
    """Inline SVG using the same path set the CSS masks are built from."""
    path = ICON_PATHS.get(name, "")
    return (
        f"<svg viewBox='0 0 24 24' width='{size}' height='{size}' fill='none' "
        "stroke='currentColor' stroke-width='1.8' stroke-linecap='round' "
        f"stroke-linejoin='round'><path d='{path}'/></svg>"
    )


def _short(model: str) -> str:
    return (model or "—").split("/")[-1]


def brand_html() -> str:
    return (
        "<div class='fb-brand'>"
        f"<div class='fb-brand-mark'>{icon('doc', 18)}</div>"
        "<div><div class='fb-brand-name'>文档助手</div>"
        f"<div class='fb-brand-sub'>v{VERSION} · 硅基流动</div></div>"
        "</div>"
    )


def rail_status_html(service: Service) -> str:
    """Four dots, not a table: whether each model is actually reachable."""
    info = service.system_info()
    ocr = info["ocr"]
    rows = [
        (info["llm"]["configured"], "对话", _short(info["llm"]["model"])),
        (info["embedding"]["configured"], "向量", _short(info["embedding"]["model"])),
        (ocr["enabled"], "视觉", _short(ocr["model"])),
        (info["rerank"]["configured"], "精排", _short(info["rerank"]["model"])),
    ]
    cells = "".join(
        f"<div class='fb-status-row'><i class='fb-dot{'' if ok else ' off'}'></i>"
        f"<span>{label}</span><b>{html.escape(name)}</b></div>"
        for ok, label, name in rows
    )
    return f"<div class='fb-status'>{cells}</div>"


def pro_hint_html() -> str:
    return (
        "<div class='fb-pro-hint'>"
        f"<span class='fb-hint-icon'>{icon('lock', 13)}</span>"
        "<div><b>页级诊断</b>与<b>检索指标</b>属于专业视图。"
        "切到增强检索后解锁完整流水线、精排日志与评测。</div></div>"
    )


def page_head_html(title: str, desc: str) -> str:
    return (
        "<div class='fb-page-head'>"
        f"<div class='fb-page-title'>{html.escape(title)}</div>"
        f"<div class='fb-page-desc'>{html.escape(desc)}</div></div>"
    )


def chat_head_html() -> str:
    return (
        "<div class='fb-chat-head'>"
        f"<div class='fb-chat-avatar'>{icon('chat', 18)}</div>"
        "<div><div class='fb-chat-name'>文档问答<span class='fb-live'>在线</span></div>"
        "<div class='fb-chat-meta'>回答只依据命中的块，每个结论都带引用编号</div></div>"
        "</div>"
    )


def mode_badge_html(mode: str) -> str:
    if mode == "enhanced":
        return (
            "<span class='fb-badge blue'>增强检索</span>"
            "<div class='fb-badge-note'>改写 + 精排已启用</div>"
        )
    return (
        "<span class='fb-badge'>快速检索</span>"
        "<div class='fb-badge-note'>两路并行，直接作答</div>"
    )


def composer_hint_html() -> str:
    return (
        "<div class='fb-composer-hint'>"
        "<kbd>Enter</kbd> 发送　<kbd>Shift</kbd>+<kbd>Enter</kbd> 换行　"
        "<span class='fb-sep'>·</span> 范围留空即检索全部文档"
        "</div>"
    )


NAV_ITEMS = [
    ("chat", "对话"),
    ("docs", "文档与解析"),
    ("chunks", "分块地图"),
    ("pages", "页级诊断"),
    ("text", "原文核对"),
    ("metrics", "检索指标"),
    ("system", "运行状态"),
]
PRO_PANELS = {"pages", "metrics"}


THEME = gr.themes.Base(
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
    primary_hue="blue",
    neutral_hue="gray",
    radius_size=gr.themes.sizes.radius_sm,
    spacing_size=gr.themes.sizes.spacing_sm,
)


def build_ui(service: Service) -> gr.Blocks:
    vision_note = settings.vision_state[1]

    # Gradio 6 takes theme/css/head at mount time, not at Blocks() time. The
    # app mounts through app.main.gr_mount, which passes THEME / CSS / HEAD.
    with gr.Blocks(title=f"文档助手 v{VERSION}") as demo:
        doc_state = gr.State(None)
        page_state = gr.State([])
        chunk_state = gr.State([])
        source_state = gr.State([])
        history_state = gr.State([])
        gen_state = gr.State([])

        with gr.Row(elem_id="fb-shell"):
            # ───────────────────────── left rail ─────────────────────────
            with gr.Column(scale=0, min_width=268, elem_id="fb-rail"):
                gr.HTML(brand_html())
                gr.HTML("<div class='fb-cap'>检索强度</div>")
                mode = gr.Radio(
                    choices=MODE_CHOICES, value="fast", show_label=False,
                    container=False, elem_id="fb-mode",
                )
                gr.HTML("<div class='fb-cap fb-cap-row'>工作区</div>")
                with gr.Column(elem_id="fb-nav"):
                    nav_chat = gr.Button("对话", variant="primary", elem_id="nav-chat")
                    nav_docs = gr.Button("文档与解析", elem_id="nav-docs")
                    nav_chunks = gr.Button("分块地图", elem_id="nav-chunks")
                    nav_pages = gr.Button("页级诊断", elem_id="nav-pages", elem_classes=["fb-pro"])
                    nav_text = gr.Button("原文核对", elem_id="nav-text")
                    nav_metrics = gr.Button("检索指标", elem_id="nav-metrics", elem_classes=["fb-pro"])
                    nav_system = gr.Button("运行状态", elem_id="nav-system")
                gr.HTML(pro_hint_html())
                gr.HTML(rail_status_html(service))

            # ───────────────────────── main column ─────────────────────────
            with gr.Column(scale=1, elem_id="fb-main"):
                flow = gr.HTML(flow_html(step_states(None, [])), elem_id="fb-flowbar")

                # ---- 对话 ----
                with gr.Column(visible=True, elem_id="panel-chat") as panel_chat:
                    with gr.Row(elem_id="fb-chatpage"):
                        with gr.Column(scale=1, elem_id="fb-chatcard"):
                            with gr.Row(elem_id="fb-chat-headrow"):
                                gr.HTML(chat_head_html())
                                chat_scope = gr.Dropdown(
                                    choices=_doc_choices(service), value=[], multiselect=True,
                                    show_label=True, label="检索范围（留空＝全部文档）",
                                    elem_id="fb-scope",
                                )
                                mode_badge = gr.HTML(mode_badge_html("fast"))
                            # 空库时这里就是入口：不摆一个上传框，用户不知道对话从哪开始。
                            with gr.Row(elem_id="fb-chat-upload"):
                                chat_upload = gr.File(
                                    label="先把文档放进来：PDF / Word / PPT / Excel / Markdown，可多选",
                                    file_types=list(SUPPORTED_EXT),
                                    file_count="multiple",
                                    elem_id="fb-chat-drop",
                                )
                                chat_upload_btn = gr.Button(
                                    "上传并入库", variant="primary", elem_id="fb-chat-upload-btn"
                                )
                            chat_note = gr.Markdown(elem_id="fb-chat-note")
                            chatbot = gr.Chatbot(
                                height=460, layout="bubble", show_label=False,
                                container=False, group_consecutive_messages=False,
                                elem_id="fb-chat",
                                placeholder=(
                                    "先上传文档，或直接提问。\n"
                                    "试试：付款账期最长是多少天？· 供应商准入需要什么材料？"
                                ),
                            )
                            answer_meta = gr.Markdown(elem_id="fb-answer-meta")
                            with gr.Row(elem_id="fb-composer"):
                                chat_input = gr.Textbox(
                                    show_label=False, container=False, lines=1, max_lines=5,
                                    placeholder="问点什么，例如：付款账期最长是多少天？",
                                    elem_id="fb-input",
                                )
                                send_btn = gr.Button("发送", elem_id="fb-send")
                                clear_btn = gr.Button("清空", elem_id="fb-clear")
                            gr.HTML(composer_hint_html())

                        with gr.Column(scale=0, min_width=340, elem_id="fb-side"):
                            trace = gr.HTML(render_trace(plan_steps("fast"), "fast"))
                            hit_map = gr.HTML(chunk_map_html([], title="命中块"))
                            with gr.Accordion("命中列表与原文", open=False):
                                source_table = gr.Dataframe(
                                    headers=SOURCE_HEADERS, interactive=False, wrap=True,
                                    label="命中列表",
                                )
                                with gr.Row():
                                    source_dd = gr.Dropdown(label="展开原文", choices=[], value=None, scale=3)
                                    source_box = gr.Textbox(label="块原文", lines=8, max_lines=24, scale=9)
                            with gr.Accordion("要不要用增强检索？", open=False):
                                compare_btn = gr.Button("对比两种模式给这个问题")
                                compare_md = gr.Markdown(
                                    "上一次提问之后点这里：两种模式各检索一次（不调用生成模型），"
                                    "直接看增强多带回了哪些页、又挤掉了哪些页。"
                                )

                # ---- 文档与解析 ----
                with gr.Column(visible=False, elem_id="panel-docs") as panel_docs:
                    gr.HTML(page_head_html(
                        "文档与解析",
                        "上传后立刻解析：先看清它被认成什么格式、每页有多少字，再决定要不要建索引。",
                    ))
                    with gr.Row(elem_id="fb-docs-grid"):
                        with gr.Column(scale=4):
                            upload = gr.File(
                                label="上传文件（可多选）",
                                file_types=list(SUPPORTED_EXT),
                                file_count="multiple",
                            )
                            auto_index = gr.Checkbox(label="入库后自动建索引", value=True)
                            with gr.Accordion("顺便做增强识别（扫描页 OCR / 图表描述，会计费）", open=False):
                                do_enhance = gr.Checkbox(label="入库后立即增强", value=False)
                                limit_pages = gr.Slider(
                                    label="最多调用模型的页数", minimum=1, maximum=60,
                                    value=settings.ocr_max_pages, step=1,
                                )
                            upload_btn = gr.Button("解析并入库", variant="primary")
                            status = gr.Markdown(
                                "支持 PDF、Word、PPT、Excel、CSV、Markdown、TXT、HTML。"
                                f"视觉模型当前：{settings.ocr_model}（{vision_note}）。"
                            )
                        with gr.Column(scale=8):
                            # 默认只露文档列表；索引参数与解析明细收起来，
                            # 免得非专业用户一进页面就被十来个数字吓回去。
                            with gr.Accordion("文档列表", open=True):
                                doc_table = gr.Dataframe(
                                    headers=DOC_HEADERS, value=_doc_rows(service),
                                    interactive=False, wrap=True, show_label=False,
                                    elem_id="fb-doc-table",
                                )
                            with gr.Accordion("索引与维护（先选中一篇再操作）", open=False):
                                with gr.Row(elem_id="fb-toolbar"):
                                    doc_dd = gr.Dropdown(
                                        label="当前文档", choices=_doc_choices(service), value=None, scale=3
                                    )
                                    chunk_size = gr.Slider(
                                        label="块长上限（字）", minimum=200, maximum=1200,
                                        value=settings.chunk_size, step=50, scale=4,
                                    )
                                    reindex_btn = gr.Button("重建索引", scale=1)
                                    delete_btn = gr.Button("删除", variant="stop", scale=1)
                            with gr.Accordion("解析明细（页数、字符量、路由依据）", open=False):
                                with gr.Row(elem_id="fb-parse-grid"):
                                    with gr.Column():
                                        parse_summary = gr.Markdown("选择文档后显示。")
                                    with gr.Column():
                                        parse_strip = gr.HTML("")
                                parse_note = gr.Markdown("")

                # ---- 分块地图 ----
                with gr.Column(visible=False, elem_id="panel-chunks") as panel_chunks:
                    gr.HTML(page_head_html(
                        "分块地图",
                        "每个块都是检索的最小单位。条形长度＝字数，颜色＝块类型；"
                        "问答后被引用的块会标红并挂上引用序号。",
                    ))
                    chunk_summary = gr.Markdown()
                    chunk_map = gr.HTML(chunk_map_html([]))
                    with gr.Row(elem_id="fb-chunk-grid"):
                        chunk_dd = gr.Dropdown(label="查看块", choices=[], value=None, scale=3)
                        chunk_box = gr.Textbox(label="块全文", lines=8, max_lines=24, scale=9)

                # ---- 页级诊断（专业）----
                with gr.Column(visible=False, elem_id="panel-pages") as panel_pages:
                    gr.HTML(page_head_html(
                        "页级诊断",
                        "扫描页走 OCR、表格页走精排、图形页走图注模型——这张表就是路由依据。",
                    ))
                    page_table = gr.Dataframe(
                        headers=PAGE_HEADERS, interactive=False, wrap=True,
                        label="每页的文本量与结构信号",
                    )

                # ---- 原文核对 ----
                with gr.Column(visible=False, elem_id="panel-text") as panel_text:
                    gr.HTML(page_head_html(
                        "原文核对",
                        "逐页对照识别结果。哪一页读错了，问题就出在这里。",
                    ))
                    with gr.Row(elem_id="fb-text-grid"):
                        view_dd = gr.Dropdown(label="页码", choices=[], value=None, scale=2)
                        page_box = gr.Textbox(label="单页原文", lines=18, max_lines=40, scale=10)

                # ---- 检索指标（专业）----
                with gr.Column(visible=False, elem_id="panel-metrics") as panel_metrics:
                    gr.HTML(page_head_html(
                        "检索指标",
                        "在这份文档上分别跑两种模式，只做检索、不调用生成模型。"
                        "命中率回答「找得到吗」，MRR 回答「排得靠前吗」。"
                        "题目可以自动出：从每个块里生成一道只能靠该页回答的问题，页码就是标准答案。",
                    ))
                    with gr.Row(elem_id="fb-metrics-bar"):
                        gen_btn = gr.Button("自动出题并评测", variant="primary", scale=2)
                        gen_count = gr.Slider(
                            label="题目数", minimum=3, maximum=20, value=8, step=1, scale=3
                        )
                        metrics_scope = gr.Dropdown(
                            label="文档范围（留空 = 全部）",
                            choices=_doc_choices(service), multiselect=True, value=[], scale=4,
                        )
                    gen_note = gr.Markdown(elem_id="fb-gen-note")
                    metrics_btn = gr.Button("只跑 golden.jsonl（示例题集）", scale=1)
                    metrics_md = gr.Markdown(
                        "还没有跑过。示例库上的参考值：快速 MRR 0.972、增强 MRR 1.000（21 题）。"
                    )

                # ---- 运行状态 ----
                with gr.Column(visible=False, elem_id="panel-system") as panel_system:
                    gr.HTML(page_head_html(
                        "运行状态",
                        "模型是否就绪、索引有多大、这个会话花了多少 token，都在这里。",
                    ))
                    with gr.Row(elem_id="fb-keybar"):
                        key_box = gr.Textbox(
                            label="临时模型 Key（只存内存，不落盘；重启失效）",
                            type="password", placeholder="sk-...", scale=8,
                        )
                        key_btn = gr.Button("应用", scale=1)
                        sys_btn = gr.Button("刷新状态", scale=1)
                    key_md = gr.Markdown(
                        "克隆仓库后没有 `.env` 也能用：在这里贴一次 Key 即可。"
                        "问答、向量、精排、视觉共用这一把，长期使用建议写进 `.env` 的 `SILICONFLOW_API_KEY`。"
                    )
                    sys_md = gr.Markdown(_system_markdown(service))

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
            return render_trace(plan_steps(mode_value), mode_value), mode_badge_html(mode_value)

        def on_nav(target):
            """Switch panels and move the highlight. Purely presentational."""
            return (
                [_update(variant="primary" if key == target else "secondary") for key, _ in NAV_ITEMS]
                + [_update(visible=(key == target)) for key, _ in NAV_ITEMS]
            )

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
                    chunk_map_html(pool, sources, title="命中块") if (show_map and sources) else chunk_map_html([]),
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
                            meta_md, _source_rows(sources), chunk_map_html(pool, sources, title="命中块"), history,
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
                            chunk_map_html(pool, sources, title="命中块"),
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
                    _source_rows(sources), chunk_map_html(pool, sources, title="命中块"), history,
                    sources, _update(choices=_source_choices(sources), value=None),
                )

        def on_clear(mode_value):
            return (
                [], "", render_trace(plan_steps(mode_value), mode_value), "", [],
                chunk_map_html([], title="命中块"),
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

        def _both_modes(doc_ids, questions, progress, base=0.0):
            """两种模式各跑一次，共用同一批题目。返回 (markdown, 错误信息)。"""
            labels = {"fast": MODE_CHOICES[0][0], "enhanced": MODE_CHOICES[1][0]}
            parts: list[str] = []
            for index, mode_name in enumerate(("fast", "enhanced"), start=1):
                progress(base + (1 - base) * index / 2, desc=f"评测：{labels[mode_name]}")
                try:
                    result = service.evaluate(
                        top_k=5, mode=mode_name, doc_ids=doc_ids, questions=questions or None
                    )
                except Exception as exc:
                    return "", f"评测失败：{exc}"
                parts.append(f"**{labels[mode_name]}**　{eval_markdown(result)}")
            return "\n\n".join(parts), ""

        def on_metrics(scope, questions=None, progress=gr.Progress()):
            body, error = _both_modes(scope or None, questions, progress, base=0.1)
            if error:
                return error
            progress(1.0, desc="完成")
            return (
                body
                + "\n\n> 两组数字用同一批题目、同一个 top_k，只有候选池宽度和精排开关不同，"
                "所以差值可以直接读成「增强换来了什么」。"
            )

        def on_generate(scope, count, progress=gr.Progress()):
            """出题 → 立刻用同一批题跑两种模式。没有人工标注也能看到自己文档的真实数字。"""
            doc_ids = scope or None
            try:
                bundle = service.generate_questions(
                    doc_ids=doc_ids,
                    limit=int(count),
                    on_progress=lambda index, total, name: progress(
                        index / max(total, 1) * 0.55, desc=f"出题 {index}/{total}：{name}"
                    ),
                )
            except ServiceError as exc:
                return f"出题失败：{exc}", "", []
            except Exception as exc:
                return f"出题失败：{exc}", "", []

            questions = bundle["questions"]
            if not questions:
                return (
                    "**一道题也没生成。** 模型没有返回可解析的 JSON——常见原因是 Key 无效（401）"
                    "或额度不足。先去「运行状态」确认 Key，再回来点一次。",
                    "", [],
                )

            pages = sorted({page for item in questions for page in item["pages"]})
            note = (
                f"已从本文档生成 **{len(questions)} 道题**，覆盖第 {format_pages(pages) or '—'} 页"
                f"（{bundle['sampled']} 个块取样，{bundle['failed']} 个块没出成题）。\n\n"
                + "\n".join(
                    f"{index}. {item['q']}　<span class='fb-muted'>p.{item['pages'][0]}</span>"
                    for index, item in enumerate(questions, start=1)
                )
            )

            body, error = _both_modes(doc_ids, questions, progress, base=0.55)
            if error:
                return note, error, questions
            progress(1.0, desc="完成")
            return (
                note,
                body
                + "\n\n> 标准答案是「这道题出自哪一页」，不是人写的参考答案——"
                "所以命中率只说明检索有没有把那一页捞回来，不评价回答写得好不好。",
                questions,
            )

        # ================= bindings =================

        def on_chat_upload(files, current, progress=gr.Progress()):
            """首页那排上传：默认就建索引，不需要用户懂"入库/索引"两步。"""
            result = on_upload(files, True, False, settings.ocr_max_pages, current, progress)
            return (*result, result[0], _update(value=None))

        chat_upload_btn.click(
            fn=on_chat_upload,
            inputs=[chat_upload, doc_state],
            outputs=[
                status, doc_table, doc_dd, doc_state, chunk_state, chunk_map,
                chunk_summary, chunk_dd, chat_scope, flow, chat_note, chat_upload,
            ],
            show_progress=True,
        )

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
        mode.change(fn=on_mode_change, inputs=[mode], outputs=[trace, mode_badge])

        _nav_buttons = [nav_chat, nav_docs, nav_chunks, nav_pages, nav_text, nav_metrics, nav_system]
        _nav_panels = [panel_chat, panel_docs, panel_chunks, panel_pages,
                       panel_text, panel_metrics, panel_system]
        for (key, _), button in zip(NAV_ITEMS, _nav_buttons):
            button.click(
                fn=lambda target=key: on_nav(target),
                inputs=None,
                outputs=_nav_buttons + _nav_panels,
            )
        metrics_btn.click(
            fn=on_metrics, inputs=[metrics_scope, gen_state], outputs=metrics_md, show_progress=True
        )
        gen_btn.click(
            fn=on_generate,
            inputs=[metrics_scope, gen_count],
            outputs=[gen_note, metrics_md, gen_state],
            show_progress=True,
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
            # 旧 Key 触发的熔断对新 Key 不成立，否则换完 Key 还得等 5 分钟
            was_paused = breaker_state()[0]
            reset_breaker()
            reset_diagnosis()      # 旧的站点判断也可能已经不成立了
            return (
                f"已应用 `{value[:6]}…{value[-4:]}`：问答、向量、精排、视觉都改用它。"
                + ("已解除向量化的熔断，可以直接重试。" if was_paused else "")
                + "重启后失效，若要长期生效请写进 `.env`。",
                _system_markdown(service),
            )

        key_btn.click(fn=on_key, inputs=[key_box], outputs=[key_md, sys_md])

    return demo
