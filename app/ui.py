"""Gradio UI: document, QA, enhancement and chunking tabs. Rendering only."""

import logging
import time

import gradio as gr

from app.state import DocStore
from rag.chunk.profile import profile_document
from rag.chunk.registry import LABELS, chunk_document, compare_strategies, chunk_preview
from rag.config import VERSION, settings
from rag.generate.answer import PageAnswerError, answer_by_pages
from rag.generate.llm import LLMError
from rag.ingest.enhance import all_blocks, blocks_of, enhance, plan_pages
from rag.ingest.loader import load_document
from rag.ingest.loader import page_text as page_text_view
from rag.utils import format_pages, parse_pages

log = logging.getLogger(__name__)

PAGE_HEADERS = ["页", "字符", "密度", "表格数", "扫描页", "表格页"]
ENHANCE_HEADERS = ["页", "路由", "判定依据", "状态", "原字符", "增强后", "耗时ms", "引擎", "备注"]
CHUNK_HEADERS = ["#", "页", "类型", "章节 / 条款", "字符", "预览"]


def _update(**kwargs):
    """Compatible with Gradio versions that still expose gr.update."""
    updater = getattr(gr, "update", None)
    if updater is not None:
        return updater(**kwargs)
    return gr.Dropdown(choices=kwargs.get("choices", []), value=kwargs.get("value"))


def _file_path(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)) and value:
        return _file_path(value[0])
    name = getattr(value, "name", None)
    return name if isinstance(name, str) else None


def _table(doc: dict, only_suspect: bool) -> tuple[list[list], list[int]]:
    rows, row_pages = [], []
    for page in doc["pages"]:
        suspicious = page["scan_like"] or page["table_like"]
        if only_suspect and not suspicious:
            continue
        rows.append(
            [
                page["page"],
                page["chars"],
                page["density"],
                page["tables"],
                "是" if page["scan_like"] else "",
                "是" if page["table_like"] else "",
            ]
        )
        row_pages.append(page["page"])
    return rows, row_pages


def _summary(doc: dict, elapsed_ms: int) -> str:
    scan = doc["scan_like_pages"]
    table = doc["table_like_pages"]
    return (
        f"**{doc['filename']}**　｜　{doc['page_count']} 页　｜　"
        f"{doc['chars']:,} 字符　｜　耗时 {elapsed_ms} ms\n\n"
        f"- 疑似扫描页：{len(scan)} 页　{format_pages(scan)}\n"
        f"- 疑似表格页：{len(table)} 页　{format_pages(table)}\n"
        f"- doc_id：`{doc['id']}`"
    )


def _profile_markdown(profile) -> str:
    data = profile.to_dict()
    styles = "、".join(f"{k}×{v}" for k, v in sorted(data["styles"].items(), key=lambda kv: -kv[1]))
    reasons = "\n".join(f"{i + 1}. {reason}" for i, reason in enumerate(data["reasons"]))
    return (
        f"### 推荐策略：`{data['strategy']}`（把握：{data['confidence']}）\n\n"
        f"块长 **{data['chunk_size']}** 字　｜　重叠 **{data['overlap']}** 字\n\n"
        f"{reasons}\n\n"
        "---\n\n"
        "| 结构指标 | 值 |\n| --- | --- |\n"
        f"| 字符 / 行数 / 页数 | {data['chars']:,} / {data['lines']} / {data['pages']} |\n"
        f"| 标题行数 / 占比 | {data['heading_lines']} / {data['heading_ratio']:.2%} |\n"
        f"| 最大层级 | {data['max_depth']} |\n"
        f"| 条款数（第X条） | {data['article_count']} |\n"
        f"| 编号风格 | {styles or '无'} |\n"
        f"| 规范性用语占比 | {data['modality_density']:.2%} |\n"
        f"| 平均句长 / 段落 90 分位 | {data['avg_sentence_len']} / {data['p90_paragraph_len']} 字 |\n"
        f"| 表格页 / 图形页占比 | {data['table_page_ratio']:.0%} / {data['figure_page_ratio']:.0%} |\n"
        f"| 结构强度评分 | {data['structure_score']:.2f} |\n"
    )


def _metrics_markdown(result: dict) -> str:
    metrics = result.get("metrics") or {}
    info = result.get("info") or {}
    note = ""
    if result["strategy"] == "hybrid" and info.get("semantic_inside"):
        note = (
            f"\n- 语义二次切分触发：{info.get('semantic_sections', 0)} 个超长条款"
            if info.get("semantic_sections")
            else "\n- 本次无超长条款，混合策略等同于结构化切分"
        )
    if metrics.get("boundary_applicable", True):
        boundary_line = f"- 断句越界率：{metrics.get('boundary_violation', 0):.0%}（块末不是句末的比例，越低越好）\n"
    else:
        boundary_line = (
            "- 断句越界率：**不适用**（本文件以短句/项目符号为主，"
            f"句末标点仅占 {metrics.get('prose_ratio', 0):.0%}，该项已从结构得分中剔除）\n"
        )
    return (
        f"策略 **{LABELS.get(result['strategy'], result['strategy'])}**　｜　"
        f"{metrics.get('chunks', 0)} 块　｜　平均 {metrics.get('avg_chars', 0)} 字　｜　"
        f"每千字 {metrics.get('chunks_per_1k', 0)} 块　｜　耗时 {result['ms']} ms\n\n"
        + (
            f"- 条款完整率：{metrics.get('clause_integrity', 0):.0%}（第X条未被切断的比例）\n"
            if metrics.get("clause_applicable", True)
            else "- 条款完整率：**不适用**（本文件无「第X条」编号，该项已从得分中剔除）\n"
        )
        + f"{boundary_line}"
        f"- 长度命中率：{metrics.get('size_conformance', 0):.0%}\n"
        f"- 类型保留率："
        + (
            "—（本文件无表格/图形块）"
            if metrics.get("typed_preservation") is None
            else f"{metrics['typed_preservation']:.0%}（表格/图形块保留为独立块的比例）"
        )
        + "\n"
        f"- 章节覆盖率：{metrics.get('section_coverage', 0):.0%}\n"
        f"- 结构得分：**{metrics.get('struct_score', 0):.3f}**"
        f"{note}\n\n"
        "> 以上为结构代理指标，可在毫秒内算出，用于比较切分质量；检索命中率 hit@5 / MRR "
        "需要标注问题，在 V6 引入。"
    )


def _pct(value, applicable: bool) -> str:
    if not applicable:
        return "—"
    return f"{value:.0%}"


def _compare_rows(rows: list[dict]) -> list[list]:
    out = []
    for row in rows:
        typed = row.get("typed_preservation")
        out.append(
            [
                ("★ " + row["label"]) if row["recommended"] else row["label"],
                row["chunks"],
                row["avg_chars"],
                _pct(row["clause_integrity"], row.get("clause_applicable", True)),
                _pct(row["boundary_violation"], row.get("boundary_applicable", True)),
                "—" if typed is None else f"{typed:.0%}",
                f"{row['section_coverage']:.0%}",
                f"{row['struct_score']:.3f}",
                row["ms"],
            ]
        )
    return out


def _chunk_rows(chunks: list) -> list[list]:
    return [
        [chunk.idx, f"{chunk.page_from}" if chunk.page_from == chunk.page_to else f"{chunk.page_from}-{chunk.page_to}",
         chunk.kind, chunk.section or "—", chunk.chars, chunk_preview(chunk)]
        for chunk in chunks
    ]


def build_ui(store: DocStore) -> gr.Blocks:
    key_state = "已配置" if settings.llm_configured else "未配置（问答将提示配置）"
    ocr_state = f"已启用 · {settings.ocr_model}" if settings.ocr_enabled else "未启用"
    embed_state = f"可用 · {settings.embed_model}" if settings.embed_ready else "未配置 Key"

    with gr.Blocks(title=f"rag-agent v{VERSION}") as demo:
        gr.Markdown(
            f"# rag-agent v{VERSION} · 文档理解与切分\n"
            f"问答模型 `{settings.llm_model}`　｜　API Key {key_state}　｜　"
            f"OCR {ocr_state}　｜　向量 {embed_state}\n\n"
            "V2 新增：表格/图形/扫描页的增强识别通道，以及四种切分策略与自动推荐。"
            "检索与精排仍在 V3。"
        )

        doc_state = gr.State(None)
        rows_state = gr.State([])
        extract_state = gr.State([])
        chunk_state = gr.State([])

        with gr.Tab("文档"):
            with gr.Row():
                with gr.Column(scale=4):
                    file_in = gr.File(
                        label="上传 PDF",
                        file_types=[".pdf"],
                        file_count="single",
                    )
                    parse_btn = gr.Button("解析", variant="primary")
                    only_suspect = gr.Checkbox(label="仅显示可疑页", value=False)
                    status = gr.Markdown("等待上传。")
                with gr.Column(scale=7):
                    table = gr.Dataframe(
                        headers=PAGE_HEADERS,
                        interactive=False,
                        wrap=True,
                        label="页级统计（点击任意行查看原文）",
                    )

            with gr.Row():
                view_dd = gr.Dropdown(label="查看页码（备用入口）", choices=[], value=None, scale=2)
                page_box = gr.Textbox(label="单页原文", lines=16, max_lines=30, scale=8)

        with gr.Tab("增强识别"):
            gr.Markdown(
                "对缺少文本层的扫描页、表格页、图形页分别走不同通道。"
                "**默认不自动运行**：调用视觉模型会计费，所以由你手动触发，并显示每一页的判定依据与耗时。\n\n"
                "- 扫描页 → OCR（`" + settings.ocr_model + "`）\n"
                "- 表格页 → pdfplumber 结构化，离线、免费\n"
                "- 图形页 → 图注（`" + settings.vision_model + "`），要求模型输出 `A -> B` 的箭头关系\n\n"
                "OCR 只能拿到图形里的文字标签，**箭头关系必须由视觉模型描述**，这是流程图能被检索到的前提。"
            )
            with gr.Row():
                enhance_btn = gr.Button("运行增强识别", variant="primary", scale=3)
                plan_btn = gr.Button("只看路由计划（不调用模型）", scale=3)
                limit_pages = gr.Slider(
                    label="最多调用模型的页数", minimum=1, maximum=50, value=settings.ocr_max_pages, step=1, scale=3
                )
            enhance_summary = gr.Markdown()
            enhance_table = gr.Dataframe(
                headers=ENHANCE_HEADERS,
                interactive=False,
                wrap=True,
                label="逐页判定与耗时",
            )
            with gr.Row():
                enhance_dd = gr.Dropdown(label="查看页码（备用入口）", choices=[], value=None, scale=2)
                enhance_box = gr.Textbox(label="增强后文本", lines=18, max_lines=40, scale=8)

        with gr.Tab("切分策略"):
            gr.Markdown(
                "四种策略共用同一份文本，差异只在切分点。**自动推荐**读取文档编号结构，"
                "不调用模型，毫秒级返回。"
            )
            with gr.Row():
                profile_btn = gr.Button("分析文档结构", scale=2)
                strategy_dd = gr.Dropdown(
                    label="策略",
                    choices=["auto"] + list(LABELS)[:4],
                    value="auto",
                    scale=3,
                )
                size_slider = gr.Slider(label="块长上限（字）", minimum=200, maximum=1200, value=500, step=50, scale=2)
                overlap_slider = gr.Slider(label="重叠（字）", minimum=0, maximum=300, value=80, step=10, scale=2)
            with gr.Row():
                chunk_btn = gr.Button("按当前策略切分", variant="primary", scale=2)
                compare_btn = gr.Button("四种策略对比", scale=2)
            profile_box = gr.Markdown()
            metrics_box = gr.Markdown()
            compare_table = gr.Dataframe(
                headers=["策略", "块数", "平均字数", "条款完整", "越界率", "类型保留", "章节覆盖", "结构得分", "耗时ms"],
                interactive=False,
                wrap=True,
                label="四种策略结构指标对比（★ 为自动推荐项）",
            )
            chunk_table = gr.Dataframe(
                headers=CHUNK_HEADERS, interactive=False, wrap=True, label="切分结果"
            )
            with gr.Row():
                chunk_dd = gr.Dropdown(label="查看块（备用入口）", choices=[], value=None, scale=2)
                chunk_box = gr.Textbox(label="块全文", lines=16, max_lines=36, scale=8)

        with gr.Tab("问答"):
            gr.Markdown(
                "V2 仍以人工选页替代检索。回答仅基于所选页面，信息不足时会明确说明。"
                "运行增强识别后，这里的单页原文自动使用增强结果。"
            )
            with gr.Row():
                page_pick = gr.Dropdown(label="选择页码", choices=[], multiselect=True, scale=6)
                pick_suspect = gr.Button("填入可疑页", scale=2)
            extra_pages = gr.Textbox(label="补充页码范围（可选，如 1-5,8）", placeholder="1-3,7")
            question = gr.Textbox(label="问题", lines=3, placeholder="请输入针对所选页面的问题…")
            ask_btn = gr.Button("提问", variant="primary")
            answer_box = gr.Markdown()

        def on_parse(file_value, show_suspect_only):
            path = _file_path(file_value)
            empty = (
                [],
                None,
                [],
                _update(choices=[], value=None),
                _update(choices=[], value=[]),
                _update(choices=[], value=None),
                _update(choices=[], value=None),
                _update(value=[]),
                _update(value=[]),
            )

            if not path:
                return ("请先上传 PDF 文件。", *empty)
            if not path.lower().endswith(".pdf"):
                return ("本版仅支持 PDF。Word / Excel / PPT 的解析将在后续版本提供。", *empty)

            started = time.perf_counter()
            try:
                doc = load_document(path)
            except ValueError as exc:
                return (f"解析失败：{exc}", *empty)

            elapsed_ms = int((time.perf_counter() - started) * 1000)
            store.put(doc)
            rows, row_pages = _table(doc, bool(show_suspect_only))
            choices = [page["page"] for page in doc["pages"]]

            log.info("parsed doc=%s pages=%d ms=%d", doc["id"], doc["page_count"], elapsed_ms)
            return (
                _summary(doc, elapsed_ms),
                rows,
                doc["id"],
                row_pages,
                _update(choices=choices, value=None),
                _update(choices=choices, value=[]),
                _update(choices=choices, value=None),
                _update(value=[]),
                _update(value=[]),
            )

        def on_filter(show_suspect_only, doc_id):
            doc = store.get(doc_id)
            if doc is None:
                return [], []
            return _table(doc, bool(show_suspect_only))

        def on_select(evt: gr.SelectData, doc_id, row_pages):
            doc = store.get(doc_id)
            if doc is None:
                return "请先在左侧解析一份 PDF。"
            raw = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
            if raw is None or not isinstance(raw, int) or raw >= len(row_pages):
                return "未选中有效行。"
            return page_text_view(doc, row_pages[raw])

        def on_view(page_no, doc_id):
            doc = store.get(doc_id)
            if doc is None:
                return "请先解析文档。"
            if not page_no:
                return "请选择一个页码。"
            return page_text_view(doc, int(page_no))

        def on_fill_suspect(doc_id):
            doc = store.get(doc_id)
            if doc is None:
                return _update(choices=[], value=[])
            pages = sorted(set(doc["scan_like_pages"]) | set(doc["table_like_pages"]))
            return _update(choices=[page["page"] for page in doc["pages"]], value=pages)

        # ---- enhancement ----

        def on_plan(doc_id):
            doc = store.get(doc_id)
            if doc is None:
                return "请先在「文档」页解析一份 PDF。", _update(value=[]), [], _update(choices=[], value=None)
            plans = plan_pages(doc)
            rows = [
                [p.page, " + ".join(p.routes), p.reason[:60], "计划中", p.chars, "", "", "—", ""]
                for p in plans
            ]
            pages = [p.page for p in plans]
            summary = (
                "### 仅路由计划（未调用任何模型）\n\n"
                f"{_plan_summary(doc, plans)}\n\n"
                "点击「运行增强识别」才会真正调用视觉模型。"
            )
            return summary, rows, pages, _update(choices=pages, value=None)

        def on_enhance(doc_id, limit, progress=gr.Progress()):
            doc = store.get(doc_id)
            if doc is None:
                return "请先在「文档」页解析一份 PDF。", [], [], _update(choices=[], value=None)
            if not settings.ocr_enabled:
                return (
                    "OCR 未启用：请在 `.env` 中设置 `OCR_ENABLED=true` 后重启服务。"
                    "表格页通道不依赖 OCR，可先在「切分策略」页看到效果。",
                    [],
                    [],
                    _update(choices=[], value=None),
                )

            total = max(min(int(limit), doc["page_count"]), 1)
            done = {"n": 0}

            def on_event(event):
                if event.get("stage") == "page_start":
                    done["n"] += 1
                    routes = "+".join(event.get("routes", []))
                    progress(
                        min(done["n"] / total, 1.0),
                        desc=f"第 {event['page']} 页 → {routes}",
                    )
                elif event.get("stage") == "page_done" and event.get("error"):
                    progress(desc=f"第 {event['page']} 页失败：{event['error'][:40]}")

            report = enhance(doc, limit_pages=int(limit), on_event=on_event)
            rows, pages = _enhance_rows(doc, report)
            summary = (
                "### 增强识别完成\n\n"
                f"{_route_summary(doc, report)}\n\n"
                f"- 调用视觉模型：**{report['api_calls']}** 次\n"
                f"- 文本量：{report['text_before']:,} → {report['text_after']:,} 字符\n"
                f"- 总耗时：{report['ms'] / 1000:.1f} s"
                + (
                    f"\n- 因预算跳过：{format_pages(report['skipped'])}"
                    if report["skipped"]
                    else ""
                )
            )
            return summary, rows, pages, _update(choices=pages, value=None)

        def on_enhance_select(evt: gr.SelectData, doc_id, extract_pages):
            doc = store.get(doc_id)
            if doc is None:
                return "请先解析文档。"
            raw = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
            if raw is None or not isinstance(raw, int) or raw >= len(extract_pages):
                return "未选中有效行。"
            return on_enhance_view(extract_pages[raw], doc_id)

        def on_enhance_view(page_no, doc_id):
            doc = store.get(doc_id)
            if doc is None:
                return "请先解析文档。"
            if not page_no:
                return "请选择一个页码。"
            blocks = blocks_of(doc, int(page_no))
            if not blocks:
                return f"第 {page_no} 页没有可显示的文本。"
            parts = []
            for block in blocks:
                label = {"text": "原文", "ocr": "OCR", "table": "表格", "figure": "图注"}.get(
                    block.type, block.type
                )
                parts.append(f"[{label} · {block.source} · {len(block.text)} 字]\n{block.text}")
            return "\n\n" + ("\n\n" + "-" * 48 + "\n\n").join(parts)

        # ---- chunking ----

        def on_profile(doc_id):
            doc = store.get(doc_id)
            if doc is None:
                return "请先在「文档」页解析一份 PDF。"
            profile = profile_document(
                all_blocks(doc),
                pages=doc["page_count"],
                table_pages=len(doc["table_like_pages"]),
            )
            return _profile_markdown(profile)

        def on_chunk(doc_id, strategy, size, overlap):
            doc = store.get(doc_id)
            empty = ("请先在「文档」页解析一份 PDF。", [], [], _update(choices=[], value=None))
            if doc is None:
                return empty
            result = chunk_document(
                all_blocks(doc),
                strategy=strategy,
                size=int(size),
                overlap=int(overlap),
                min_chars=settings.chunk_min_chars,
                percentile=settings.semantic_percentile,
                pages=doc["page_count"],
                table_pages=len(doc["table_like_pages"]),
            )
            rows = _chunk_rows(result["chunks"])
            choices = [(f"#{c.idx} · {c.chars} 字 · {c.kind}", c.idx) for c in result["chunks"]]
            return (
                _metrics_markdown(result),
                rows,
                result["chunks"],
                _update(choices=choices, value=None),
            )

        def on_compare(doc_id, size, overlap):
            doc = store.get(doc_id)
            if doc is None:
                return "请先在「文档」页解析一份 PDF。", []
            rows = compare_strategies(
                all_blocks(doc),
                size=int(size),
                overlap=int(overlap),
                min_chars=settings.chunk_min_chars,
                percentile=settings.semantic_percentile,
                pages=doc["page_count"],
                table_pages=len(doc["table_like_pages"]),
            )
            return _compare_note(rows), _compare_rows(rows)

        def on_chunk_view(chunk_idx, chunks):
            if not chunks:
                return "请先切分文档。"
            if chunk_idx is None:
                return "请选择一个块。"
            for chunk in chunks:
                if chunk.idx == int(chunk_idx):
                    return (
                        f"[#{chunk.idx} · 第 {chunk.page_from}-{chunk.page_to} 页 · "
                        f"{chunk.kind} · {chunk.chars} 字]\n"
                        f"章节：{chunk.section or '—'}\n\n{chunk.text}"
                    )
            return "未找到该块。"

        def on_ask(doc_id, picked, extra, question_text):
            doc = store.get(doc_id)
            if doc is None:
                return "请先在「文档」页解析一份 PDF。"

            pages = sorted({int(p) for p in (picked or [])} | set(parse_pages(extra)))
            if not pages:
                return "请至少选择一页：可在下拉框多选，或在「补充页码范围」中填写，如 `1-5,8`。"

            try:
                result = answer_by_pages(doc, pages, question_text)
            except PageAnswerError as exc:
                return f"提示：{exc}"
            except LLMError as exc:
                return f"调用模型失败：{exc}"

            usage = ""
            if result["prompt_tokens"] or result["completion_tokens"]:
                usage = f"　｜　tokens {result['prompt_tokens']} / {result['completion_tokens']}"
            return (
                f"{result['answer']}\n\n"
                "---\n"
                f"来源：第 {format_pages(result['pages'])} 页　｜　"
                f"上下文 {result['chars']:,} 字符{usage}"
            )

        parse_btn.click(
            fn=on_parse,
            inputs=[file_in, only_suspect],
            outputs=[
                status, table, doc_state, rows_state, view_dd, page_pick,
                enhance_dd, chunk_dd, compare_table, chunk_table,
            ],
            show_progress=True,
        )
        only_suspect.change(fn=on_filter, inputs=[only_suspect, doc_state], outputs=[table, rows_state])
        table.select(fn=on_select, inputs=[doc_state, rows_state], outputs=page_box)
        view_dd.change(fn=on_view, inputs=[view_dd, doc_state], outputs=page_box)
        pick_suspect.click(fn=on_fill_suspect, inputs=[doc_state], outputs=page_pick)

        plan_btn.click(
            fn=on_plan,
            inputs=[doc_state],
            outputs=[enhance_summary, enhance_table, extract_state, enhance_dd],
            show_progress=True,
        )
        enhance_btn.click(
            fn=on_enhance,
            inputs=[doc_state, limit_pages],
            outputs=[enhance_summary, enhance_table, extract_state, enhance_dd],
            show_progress=True,
        )
        enhance_dd.change(fn=on_enhance_view, inputs=[enhance_dd, doc_state], outputs=enhance_box)
        enhance_table.select(
            fn=on_enhance_select,
            inputs=[doc_state, extract_state],
            outputs=enhance_box,
        )

        profile_btn.click(fn=on_profile, inputs=[doc_state], outputs=profile_box, show_progress=True)
        chunk_btn.click(
            fn=on_chunk,
            inputs=[doc_state, strategy_dd, size_slider, overlap_slider],
            outputs=[metrics_box, chunk_table, chunk_state, chunk_dd],
            show_progress=True,
        )
        compare_btn.click(
            fn=on_compare,
            inputs=[doc_state, size_slider, overlap_slider],
            outputs=[metrics_box, compare_table],
            show_progress=True,
        )
        chunk_dd.change(fn=on_chunk_view, inputs=[chunk_dd, chunk_state], outputs=chunk_box)

        ask_btn.click(
            fn=on_ask,
            inputs=[doc_state, page_pick, extra_pages, question],
            outputs=answer_box,
            show_progress=True,
        )

    return demo


def _enhance_rows(doc: dict, report: dict) -> tuple[list[list], list[int]]:
    """Flatten the report into table rows plus the page numbers behind them."""
    plans = {plan.page: plan for plan in report["plans"]}
    rows: list[list] = []
    pages: list[int] = []
    for result in report["pages"]:
        plan = plans.get(result.page)
        rows.append(
            [
                result.page,
                " + ".join(result.routes),
                (plan.reason if plan else "")[:60],
                {"done": "完成", "skipped": "跳过", "error": "失败"}.get(result.status, result.status),
                result.chars_before,
                result.chars_after,
                result.ms,
                result.engine or "—",
                result.error[:60],
            ]
        )
        pages.append(result.page)
    return rows, pages


def _route_summary(doc: dict, report: dict) -> str:
    counts: dict[str, int] = {}
    for plan in report["plans"]:
        for route in plan.routes:
            counts[route] = counts.get(route, 0) + 1
    labels = {"text": "文本层", "table": "表格", "ocr": "OCR", "figure": "图注"}
    breakdown = "　｜　".join(f"{labels.get(k, k)} {v} 页" for k, v in sorted(counts.items()))
    return (
        f"{doc['page_count']} 页路由结果：{breakdown}\n\n"
        f"- 需要视觉模型的页：{sum(1 for p in report['plans'] if 'ocr' in p.routes or 'figure' in p.routes)} 页"
        f"（上限 `OCR_MAX_PAGES={settings.ocr_max_pages}`）"
    )


def _plan_summary(doc: dict, plans: list) -> str:
    counts: dict[str, int] = {}
    for plan in plans:
        for route in plan.routes:
            counts[route] = counts.get(route, 0) + 1
    labels = {"text": "文本层", "table": "表格", "ocr": "OCR", "figure": "图注"}
    breakdown = "　｜　".join(f"{labels.get(k, k)} {v} 页" for k, v in sorted(counts.items()))
    need = sum(1 for plan in plans if "ocr" in plan.routes or "figure" in plan.routes)
    return (
        f"{doc['page_count']} 页路由结果：{breakdown}\n\n"
        f"- 需要视觉模型的页：{need} 页"
        f"（上限 `OCR_MAX_PAGES={settings.ocr_max_pages}`，超出的页会直接跳过并标注）"
    )


def _compare_note(rows: list[dict]) -> str:
    winner = rows[0]
    lines = [
        f"### 推荐 **{winner['label']}**",
        "",
        f"结构得分 {winner['struct_score']:.3f}　｜　{winner['chunks']} 块　｜　"
        f"平均 {winner['avg_chars']} 字　｜　条款完整 {winner['clause_integrity']:.0%}",
        "",
        "| 策略 | 块数 | 平均字数 | 条款完整 | 越界率 | 类型保留 | 章节覆盖 | 结构得分 | 耗时ms |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        typed = row.get("typed_preservation")
        lines.append(
            f"| {'★ ' if row['recommended'] else ''}{row['label']} | {row['chunks']} | "
            f"{row['avg_chars']} | "
            f"{_pct(row['clause_integrity'], row.get('clause_applicable', True))} | "
            f"{_pct(row['boundary_violation'], row.get('boundary_applicable', True))} | "
            f"{'—' if typed is None else f'{typed:.0%}'} | "
            f"{row['section_coverage']:.0%} | {row['struct_score']:.3f} | {row['ms']} |"
        )
    lines.append("")
    lines.append(
        "> 这些是结构代理指标，不是检索精度。块数过少可能只是把整篇塞进两块，"
        "所以同时看平均字数与条款完整率。"
    )
    return "\n".join(lines)
