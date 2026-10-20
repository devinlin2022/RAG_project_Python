"""Gradio UI: document tab and page-scoped QA tab. Rendering only."""

import logging
import time

import gradio as gr

from app.state import DocStore
from rag.config import VERSION, settings
from rag.generate.answer import PageAnswerError, answer_by_pages
from rag.generate.llm import LLMError
from rag.ingest.loader import load_document
from rag.ingest.loader import page_text as render_page
from rag.utils import format_pages, parse_pages

log = logging.getLogger(__name__)

HEADERS = ["页", "字符", "密度", "表格数", "扫描页", "表格页"]


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


def build_ui(store: DocStore) -> gr.Blocks:
    key_state = "已配置" if settings.llm_configured else "未配置（问答将提示配置）"

    with gr.Blocks(title=f"rag-agent v{VERSION}") as demo:
        gr.Markdown(
            f"# rag-agent v{VERSION} · 按页抽取与页级问答\n"
            f"模型 `{settings.llm_model}`　｜　API Key {key_state}　｜　"
            "本版不含检索：问答需显式选择页码。"
        )

        doc_state = gr.State(None)
        rows_state = gr.State([])

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
                    status = gr.Markdown("等待上传。V1 仅解析 PDF。")
                with gr.Column(scale=7):
                    table = gr.Dataframe(
                        headers=HEADERS,
                        interactive=False,
                        wrap=True,
                        label="页级统计（点击任意行查看原文）",
                    )

            with gr.Row():
                view_dd = gr.Dropdown(
                    label="查看页码（备用入口）",
                    choices=[],
                    value=None,
                    scale=2,
                )
                page_box = gr.Textbox(
                    label="单页原文",
                    lines=16,
                    max_lines=30,
                    scale=8,
                )

        with gr.Tab("问答"):
            gr.Markdown(
                "V1 以人工选页替代检索。回答仅基于所选页面，信息不足时会明确说明。"
            )
            with gr.Row():
                page_pick = gr.Dropdown(
                    label="选择页码",
                    choices=[],
                    multiselect=True,
                    scale=6,
                )
                pick_suspect = gr.Button("填入可疑页", scale=2)
            extra_pages = gr.Textbox(
                label="补充页码范围（可选，如 1-5,8）",
                placeholder="1-3,7",
            )
            question = gr.Textbox(label="问题", lines=3, placeholder="请输入针对所选页面的问题…")
            ask_btn = gr.Button("提问", variant="primary")
            answer_box = gr.Markdown()

        def on_parse(file_value, show_suspect_only):
            path = _file_path(file_value)
            empty = ([], None, [], _update(choices=[], value=None), _update(choices=[], value=[]))

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
            choices = [p["page"] for p in doc["pages"]]

            log.info("parsed doc=%s pages=%d ms=%d", doc["id"], doc["page_count"], elapsed_ms)
            return (
                _summary(doc, elapsed_ms),
                rows,
                doc["id"],
                row_pages,
                _update(choices=choices, value=None),
                _update(choices=choices, value=[]),
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
            return render_page(doc, row_pages[raw])

        def on_view(page_no, doc_id):
            doc = store.get(doc_id)
            if doc is None:
                return "请先解析文档。"
            if not page_no:
                return "请选择一个页码。"
            return render_page(doc, int(page_no))

        def on_fill_suspect(doc_id):
            doc = store.get(doc_id)
            if doc is None:
                return _update(choices=[], value=[])
            pages = sorted(set(doc["scan_like_pages"]) | set(doc["table_like_pages"]))
            return _update(choices=[p["page"] for p in doc["pages"]], value=pages)

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
                usage = (
                    f"　｜　tokens {result['prompt_tokens']} / {result['completion_tokens']}"
                )
            return (
                f"{result['answer']}\n\n"
                "---\n"
                f"来源：第 {format_pages(result['pages'])} 页　｜　"
                f"上下文 {result['chars']:,} 字符{usage}"
            )

        parse_btn.click(
            fn=on_parse,
            inputs=[file_in, only_suspect],
            outputs=[status, table, doc_state, rows_state, view_dd, page_pick],
            show_progress=True,
        )
        only_suspect.change(
            fn=on_filter,
            inputs=[only_suspect, doc_state],
            outputs=[table, rows_state],
        )
        table.select(fn=on_select, inputs=[doc_state, rows_state], outputs=page_box)
        view_dd.change(fn=on_view, inputs=[view_dd, doc_state], outputs=page_box)
        pick_suspect.click(fn=on_fill_suspect, inputs=[doc_state], outputs=page_pick)
        ask_btn.click(
            fn=on_ask,
            inputs=[doc_state, page_pick, extra_pages, question],
            outputs=answer_box,
            show_progress=True,
        )

    return demo
