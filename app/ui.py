"""Gradio UI: two things a user actually needs -- build a corpus, ask it.

Rendering and event binding only: every handler delegates to rag.service.
No retrieval option is exposed that a document reader would have to understand;
the two modes (快速 / 增强) set them on the user's behalf.
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

MODE_CHOICES = [("快速检索（便宜、直接）", "fast"), ("增强检索（改写 + 精排）", "enhanced")]

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


def chunk_map_html(chunks: list, hits: list[dict] | None = None, limit: int = 240) -> str:
    """Blocks as proportional bars: what the splitter produced, and what answered.

    A 300-chunk bar list is unreadable, so the list is capped and the page strip
    above it still shows the whole document at a glance.
    """
    if not chunks:
        return (
            "<div style='color:#57606a;font-size:13px;padding:8px 0'>"
            "还没有分块。上传文件并建立索引后，这里会显示每个块的位置与大小。</div>"
        )

    rank = {f"{item['doc_id']}:{item['page']}": item["n"] for item in (hits or [])}
    total_chars = sum(chunk.chars for chunk in chunks) or 1
    biggest = max(chunk.chars for chunk in chunks) or 1
    pages = max(chunk.page_to for chunk in chunks) or 1

    # Page strip: one cell per page, darker with more text, ringed when cited.
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
    "generate": ("生成回答", "只依据命中块作答并标注引用"),
}

_STATUS_COLOR = {
    "pending": ("#d8dee4", "#8b949e", "·"),
    "running": ("#1f6feb", "#1f6feb", "◐"),
    "done": ("#1a7f37", "#1a7f37", "✓"),
    "skip": ("#bf8700", "#bf8700", "–"),
    "error": ("#d1242f", "#d1242f", "!"),
}


def render_trace(steps: list[dict]) -> str:
    """Vertical timeline of the pipeline. Every line carries a measured number."""
    if not steps:
        return (
            "<div style='color:#57606a;font-size:13px;padding:10px 4px'>"
            "提问后这里会逐条显示后台动作：改写 → 向量化 → 双路检索 → 融合 → 精排 → 生成。"
            "</div>"
        )

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

    return (
        "<style>@keyframes rt-spin{from{transform:rotate(0)}to{transform:rotate(360deg)}}</style>"
        "<div style=\"font-family:ui-sans-serif,system-ui,'Segoe UI',sans-serif;"
        "background:#f6f8fa;border:1px solid #d8dee4;border-radius:8px;padding:10px 12px\">"
        "<div style='font-size:11px;letter-spacing:.08em;color:#8b949e;margin-bottom:4px'>"
        "PIPELINE</div>"
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


def _system_markdown(service: Service) -> str:
    info = service.system_info()
    usage = info["usage"]
    index = info["index"]
    price = (
        f"约 ¥{usage['cost']:.4f}（按配置单价估算）" if usage["priced"] else "未配置单价，仅统计 token"
    )
    return (
        "### 运行状态\n\n"
        "| 项 | 值 |\n| --- | --- |\n"
        f"| 版本 / 启动 | `{info['version']}` / {info['started_at']} |\n"
        f"| 对话模型 | `{info['llm']['model']}`（{'已配置' if info['llm']['configured'] else '**未配置**'}） |\n"
        f"| 接口地址 | `{info['llm']['base_url']}` |\n"
        f"| 向量模型 | `{info['embedding']['model']}`（{'可用' if info['embedding']['configured'] else '未配置'}，"
        f"batch {info['embedding']['batch']} × 并发 {info['embedding']['concurrency']}） |\n"
        f"| 精排模型 | `{info['rerank']['model']}`（{'可用' if info['rerank']['configured'] else '未启用'}） |\n"
        f"| 视觉模型 | `{info['ocr']['vision_model']}`（{'启用' if info['ocr']['enabled'] else '未启用'}，"
        f"并发 {info['ocr']['concurrency']}） |\n\n"
        "### 索引与用量\n\n"
        "| 项 | 值 |\n| --- | --- |\n"
        f"| 文档 / 块 / 向量 | {index['docs']} / {index['chunks']} / {index['vectors']} |\n"
        f"| 向量维度 / 词表 | {index['dim']} / {index['vocab']} |\n"
        f"| 提问次数 | {usage['requests']} |\n"
        f"| tokens（入 / 出 / 合计） | {usage['llm_in']:,} / {usage['llm_out']:,} / {usage['total_tokens']:,} |\n"
        f"| 向量化 tokens | {usage['embed']:,}（{price}） |\n"
        f"| 访问控制 | API Key {'必需' if info['security']['api_key_required'] else '未启用'}"
        f"　｜　界面登录 {'启用' if info['security']['ui_login'] else '未启用'} |\n"
    )


def build_ui(service: Service) -> gr.Blocks:
    ready = settings.llm_configured
    header = (
        f"# 文档助手 v{VERSION}\n"
        f"对话 `{settings.llm_model}`（{'就绪' if ready else '**缺少 API Key**'}）"
        f"　｜　向量 `{settings.embed_model}`"
        f"　｜　精排 `{settings.rerank_model}`"
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

        # ---------------- 文档 ----------------
        with gr.Tab("① 文档与分块"):
            with gr.Row():
                with gr.Column(scale=4):
                    upload = gr.File(
                        label="上传文件（可多选）",
                        file_types=list(SUPPORTED_EXT),
                        file_count="multiple",
                    )
                    auto_index = gr.Checkbox(label="入库后自动建索引", value=True)
                    with gr.Accordion("顺便做增强识别（扫描件 OCR / 图表描述，会计费）", open=False):
                        do_enhance = gr.Checkbox(label="入库后立即增强", value=False)
                        limit_pages = gr.Slider(
                            label="最多调用模型的页数",
                            minimum=1,
                            maximum=60,
                            value=settings.ocr_max_pages,
                            step=1,
                        )
                    upload_btn = gr.Button("解析并入库", variant="primary")
                    status = gr.Markdown("支持 PDF、Word、PPT、Excel、CSV、Markdown、TXT、HTML。")
                with gr.Column(scale=8):
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
                        delete_btn = gr.Button("删除", variant="stop", scale=1)
                        reindex_btn = gr.Button("重建索引", scale=2)

            gr.Markdown("### 分块可视化")
            gr.Markdown(
                "每个块都是一次检索的最小单位。条形长度＝字数，颜色＝块类型；"
                "上方页轴显示全文字量分布。问答后同一张图会把被引用的块标红。"
            )
            chunk_map = gr.HTML(chunk_map_html([]))
            chunk_summary = gr.Markdown()
            with gr.Row():
                chunk_dd = gr.Dropdown(label="查看块", choices=[], value=None, scale=3)
                chunk_size = gr.Slider(
                    label="块长上限（字）", minimum=200, maximum=1200,
                    value=settings.chunk_size, step=50, scale=4,
                )
                chunk_box = gr.Textbox(label="块全文", lines=6, max_lines=20, scale=9)

            with gr.Accordion("页级诊断", open=False):
                page_table = gr.Dataframe(
                    headers=PAGE_HEADERS, interactive=False, wrap=True,
                    label="每页的文本量与结构信号（用于判断是否走 OCR / 表格 / 图注）",
                )
                with gr.Row():
                    view_dd = gr.Dropdown(label="查看页码", choices=[], value=None, scale=2)
                    page_box = gr.Textbox(label="单页原文", lines=10, max_lines=24, scale=10)

            with gr.Accordion("运行状态与用量", open=False):
                with gr.Row():
                    key_box = gr.Textbox(
                        label="临时 API Key（只存内存，不落盘；重启失效）",
                        type="password",
                        placeholder="sk-...",
                        scale=8,
                    )
                    key_btn = gr.Button("应用", scale=1)
                    sys_btn = gr.Button("刷新状态", scale=2)
                key_md = gr.Markdown(
                    "克隆仓库后没有 `.env` 也能用：在这里贴一次 Key 即可。"
                    "长期使用建议写进 `.env` 的 `SILICONFLOW_API_KEY`。"
                )
                sys_md = gr.Markdown(_system_markdown(service))

        # ---------------- 问答 ----------------
        with gr.Tab("② 提问"):
            with gr.Row():
                mode = gr.Radio(
                    label="检索方式",
                    choices=MODE_CHOICES,
                    value="fast",
                    scale=5,
                )
                chat_scope = gr.Dropdown(
                    label="范围（留空=全部文档）",
                    choices=_doc_choices(service),
                    multiselect=True,
                    value=[],
                    scale=4,
                )
                chat_topk = gr.Slider(
                    label="引用块数", minimum=2, maximum=12, value=5, step=1, scale=3
                )
            gr.Markdown(
                "**快速检索**：一条查询词同时打向量与 BM25，RRF 融合后直接作答，约 1 秒出首字。\n\n"
                "**增强检索**：先改写追问、再扩大候选池并做交叉编码器精排，"
                "命中更准，但多花一次改写和一次精排的时间。"
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
                    trace = gr.HTML(render_trace([]))

            answer_meta = gr.Markdown()
            with gr.Accordion("本次命中的块", open=True):
                hit_map = gr.HTML(chunk_map_html([]))
                source_table = gr.Dataframe(
                    headers=SOURCE_HEADERS, interactive=False, wrap=True, label="命中列表"
                )
                with gr.Row():
                    source_dd = gr.Dropdown(label="展开原文", choices=[], value=None, scale=2)
                    source_box = gr.Textbox(label="块原文", lines=8, max_lines=24, scale=10)

        # ================= handlers =================

        def on_upload(files, also_index, also_enhance, page_limit, progress=gr.Progress()):
            paths = [p for p in _file_paths(files) if Path(p).suffix.lower() in SUPPORTED_EXT]
            if not paths:
                return (
                    "请先选择文件。", _doc_rows(service),
                    _update(choices=_doc_choices(service)), None, [],
                    chunk_map_html([]), "", _update(choices=[]),
                    _update(choices=_doc_choices(service), value=[]),
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

                if also_enhance and settings.ocr_enabled:
                    try:
                        report = service.enhance(doc["id"], limit_pages=int(page_limit))
                        lines.append(
                            f"  - 增强：调用模型 {report['api_calls']} 次，"
                            f"{report['text_before']:,} → {report['text_after']:,} 字符，"
                            f"{report['ms'] / 1000:.1f}s（并发 {report['concurrency']}）"
                        )
                        if report["api_calls"]:
                            service.index(doc["id"], force=True)
                    except Exception as exc:
                        lines.append(f"  - 增强失败：{exc}")

            progress(1.0, desc="完成")
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
            )

        def on_doc_change(doc_id):
            doc = service.docs.get(doc_id) if doc_id else None
            if doc is None:
                return (
                    "请选择或上传一份文件。", [], [], chunk_map_html([]), "", _update(choices=[]),
                    _update(choices=[]),
                )
            rows, pages = _page_rows(doc)
            chunks = [c for c in service.library.records if c.doc_id == doc["id"]]
            return (
                _doc_summary(doc),
                rows,
                pages,
                chunk_map_html(chunks),
                chunk_summary_md(chunks) or "尚未建立索引：点「重建索引」按当前块长切分。",
                _update(choices=[(f"#{c.idx} · {c.chars} 字 · {c.kind}", c.idx) for c in chunks], value=None),
                _update(choices=pages, value=None),
            )

        def on_delete(doc_id):
            choices = _doc_choices(service)
            if not doc_id:
                return (
                    "请先选择文档。", _doc_rows(service), _update(choices=choices), None,
                    [], chunk_map_html([]), "", _update(choices=[]),
                    _update(choices=choices, value=[]),
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
                _update(choices=choices, value=[]),
            )

        def on_reindex(doc_id, size):
            if not doc_id:
                return "请先选择文档。", _doc_rows(service), [], chunk_map_html([]), "", _update(choices=[])
            try:
                info = service.index(doc_id, size=int(size), force=True)
            except ServiceError as exc:
                return f"建索引失败：{exc}", _doc_rows(service), [], chunk_map_html([]), "", _update(choices=[])
            chunks = [c for c in service.library.records if c.doc_id == doc_id]
            note = (
                f"已重建索引：{info['chunks']} 块 / {info['dim']} 维"
                f"（{info['strategy']}，向量化 {info['embed_ms']:.0f} ms，"
                f"总耗时 {info['ms']:.0f} ms）"
            )
            if info["embed_error"]:
                note += f"\n\n> 向量化未生效：{info['embed_error']}\n> 当前只能用关键词检索。"
            return (
                note,
                _doc_rows(service),
                chunks,
                chunk_map_html(chunks),
                chunk_summary_md(chunks),
                _update(choices=[(f"#{c.idx} · {c.chars} 字 · {c.kind}", c.idx) for c in chunks], value=None),
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

        def on_chat(message, history, mode_value, scope, top_k):
            history = list(history or [])
            pool = [
                record
                for record in service.library.records
                if not scope or record.doc_id in scope
            ]
            blank = (
                history, "", render_trace([]), "", [], chunk_map_html(pool), history, [],
                _update(choices=[], value=None),
            )
            if not message or not message.strip():
                yield blank
                return
            if not service.library.records:
                history = history + [
                    {"role": "user", "content": message},
                    {"role": "assistant", "content": "索引为空：请先在「文档与分块」上传文件并建立索引。"},
                ]
                yield (
                    history, "", render_trace([]), "索引为空，无法检索。", [],
                    chunk_map_html([]), history, [], _update(choices=[], value=None),
                )
                return

            past = history[-settings.history_turns * 2 :]
            history = history + [{"role": "user", "content": message}]
            steps: dict[str, dict] = {}
            order: list[str] = []
            sources: list[dict] = []
            answer = ""
            meta_md = ""

            def step(key: str, status: str = "running", detail: str = "", meta: str = "") -> None:
                title, _ = STEP_TITLES[key]
                if key not in steps:
                    order.append(key)
                    steps[key] = {"key": key, "title": title}
                steps[key].update(status=status, detail=detail, meta=meta)

            def view(show_map: bool = True):
                return (
                    history + [{"role": "assistant", "content": answer or "…"}],
                    "",
                    render_trace([steps[k] for k in order]),
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
                    top_k=int(top_k),
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
                        step("dense", "running", f"与 {event.get('vectors', 0)} 个块向量比较余弦相似度")
                        yield view(False)

                    elif stage == "dense_done":
                        tops = "；".join(
                            f"p.{item['page']} {item['score']}" for item in event.get("top", [])
                        )
                        step("dense", "done", f"命中 {event['hits']} 块，最高相似度 {tops}", f"{event['ms']:.1f} ms")
                        yield view(False)

                    elif stage == "sparse_start":
                        terms = "、".join(event.get("terms") or []) or "（无有效词）"
                        step("sparse", "running", f"分词：{terms}")
                        yield view(False)

                    elif stage == "sparse_done":
                        tops = "；".join(
                            f"p.{item['page']} {item['score']}" for item in event.get("top", [])
                        )
                        step("sparse", "done", f"命中 {event['hits']} 块，最高分 {tops}", f"{event['ms']:.1f} ms")
                        yield view(False)

                    elif stage == "fuse_done":
                        step(
                            "fusion",
                            "done",
                            f"向量 {event['dense']} 条 + 关键词 {event['sparse']} 条"
                            f" → {event['candidates']} 个候选（k={event['k']}）",
                            f"{event['ms']:.1f} ms",
                        )
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
                            history, "", render_trace([steps[k] for k in order]), meta_md,
                            _source_rows(sources), chunk_map_html(pool, sources), history,
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
                            render_trace([steps[k] for k in order]),
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
                    history, "", render_trace([steps[k] for k in order]), meta_md,
                    _source_rows(sources), chunk_map_html(pool, sources), history,
                    sources, _update(choices=_source_choices(sources), value=None),
                )

        def on_clear():
            return (
                [], "", render_trace([]), "", [], chunk_map_html([]), [], [],
                _update(choices=[], value=None),
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

        # ================= bindings =================

        upload_btn.click(
            fn=on_upload,
            inputs=[upload, auto_index, do_enhance, limit_pages],
            outputs=[
                status, doc_table, doc_dd, doc_state, chunk_state, chunk_map,
                chunk_summary, chunk_dd, chat_scope,
            ],
            show_progress=True,
        )
        doc_dd.change(
            fn=on_doc_change,
            inputs=[doc_dd],
            outputs=[status, page_table, page_state, chunk_map, chunk_summary, chunk_dd, view_dd],
            show_progress=True,
        )
        view_dd.change(fn=on_page_view, inputs=[view_dd, doc_state], outputs=page_box)
        delete_btn.click(
            fn=on_delete,
            inputs=[doc_state],
            outputs=[
                status, doc_table, doc_dd, doc_state, chunk_state, chunk_map,
                chunk_summary, chunk_dd, chat_scope,
            ],
        )
        reindex_btn.click(
            fn=on_reindex,
            inputs=[doc_state, chunk_size],
            outputs=[status, doc_table, chunk_state, chunk_map, chunk_summary, chunk_dd],
            show_progress=True,
        )
        chunk_dd.change(fn=on_chunk_view, inputs=[chunk_dd, chunk_state], outputs=chunk_box)

        _chat_outputs = [
            chatbot, chat_input, trace, answer_meta,
            source_table, hit_map, history_state, source_state, source_dd,
        ]
        send_btn.click(
            fn=on_chat,
            inputs=[chat_input, history_state, mode, chat_scope, chat_topk],
            outputs=_chat_outputs,
            show_progress="hidden",
        )
        chat_input.submit(
            fn=on_chat,
            inputs=[chat_input, history_state, mode, chat_scope, chat_topk],
            outputs=_chat_outputs,
            show_progress="hidden",
        )
        clear_btn.click(fn=on_clear, inputs=[], outputs=_chat_outputs)
        source_dd.change(fn=on_source_view, inputs=[source_dd, source_state], outputs=source_box)
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
