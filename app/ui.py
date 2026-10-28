"""Gradio UI: documents, chunking, retrieval debug, chat, evaluation, system.

Rendering and event binding only: every handler delegates to rag.service.
"""

import logging
import time
from pathlib import Path

import gradio as gr

from rag.chunk.profile import profile_document
from rag.chunk.registry import LABELS, chunk_document, compare_strategies, chunk_preview
from rag.config import BASE_DIR, VERSION, settings
from rag.ingest.enhance import all_blocks, blocks_of, plan_pages
from rag.ingest.loader import page_text as page_text_view
from rag.service import Service, ServiceError
from rag.utils import format_pages

log = logging.getLogger(__name__)

PAGE_HEADERS = ["页", "字符", "密度", "表格数", "扫描页", "表格页", "图形页"]
ENHANCE_HEADERS = ["页", "路由", "判定依据", "状态", "原字符", "增强后", "耗时ms", "引擎", "备注"]
CHUNK_HEADERS = ["#", "页", "类型", "章节 / 条款", "字符", "预览"]
DOC_HEADERS = ["ID", "文件名", "页数", "字符", "索引块", "策略", "已增强", "入库时间"]
SOURCE_HEADERS = ["#", "文件", "页", "类型", "融合分", "向量", "BM25", "精排", "预览"]
EVAL_HEADERS = ["#", "问题", "命中页", "标注页", "hit", "RR", "耗时ms"]


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
            row["pages"],
            row["chars"],
            row["chunks"],
            row["strategy"],
            "是" if row["enhanced"] else "",
            row["created_at"],
        ]
        for row in service.list_documents()
    ]


def _page_rows(doc: dict, only_suspect: bool = False) -> tuple[list[list], list[int]]:
    rows, pages = [], []
    for page in doc["pages"]:
        suspicious = page["scan_like"] or page["table_like"] or page["figure_like"]
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
                "是" if page["figure_like"] else "",
            ]
        )
        pages.append(page["page"])
    return rows, pages


def _doc_summary(doc: dict, elapsed_ms: int = 0) -> str:
    tail = f"　｜　解析 {elapsed_ms} ms" if elapsed_ms else ""
    return (
        f"**{doc['filename']}**　｜　{doc['page_count']} 页　｜　{doc['chars']:,} 字符{tail}\n\n"
        f"- 扫描页 {len(doc['scan_like_pages'])}：{format_pages(doc['scan_like_pages'])}\n"
        f"- 表格页 {len(doc['table_like_pages'])}：{format_pages(doc['table_like_pages'])}\n"
        f"- 图形页 {len(doc['figure_like_pages'])}：{format_pages(doc['figure_like_pages'])}\n"
        f"- doc_id：`{doc['id']}`"
        f"　｜　索引：{doc.get('chunk_count', 0)} 块 / {doc.get('dim', 0)} 维"
        f"　｜　增强：{'已完成' if doc.get('enhanced_at') else '未运行'}"
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
        boundary_line = f"- 断句越界率：{metrics.get('boundary_violation', 0):.0%}（越低越好）\n"
    else:
        boundary_line = (
            "- 断句越界率：**不适用**（句末标点仅占 "
            f"{metrics.get('prose_ratio', 0):.0%}，已从得分中剔除）\n"
        )
    return (
        f"策略 **{LABELS.get(result['strategy'], result['strategy'])}**　｜　"
        f"{metrics.get('chunks', 0)} 块　｜　平均 {metrics.get('avg_chars', 0)} 字　｜　"
        f"每千字 {metrics.get('chunks_per_1k', 0)} 块　｜　耗时 {result['ms']} ms\n\n"
        + (
            f"- 条款完整率：{metrics.get('clause_integrity', 0):.0%}\n"
            if metrics.get("clause_applicable", True)
            else "- 条款完整率：**不适用**（无「第X条」编号）\n"
        )
        + boundary_line
        + f"- 长度命中率：{metrics.get('size_conformance', 0):.0%}\n"
        f"- 类型保留率："
        + (
            "—（本文件无表格/图形块）"
            if metrics.get("typed_preservation") is None
            else f"{metrics['typed_preservation']:.0%}"
        )
        + f"\n- 章节覆盖率：{metrics.get('section_coverage', 0):.0%}\n"
        f"- 结构得分：**{metrics.get('struct_score', 0):.3f}**{note}"
    )


def _pct(value, applicable: bool) -> str:
    return "—" if not applicable else f"{value:.0%}"


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


def _compare_note(rows: list[dict]) -> str:
    winner = rows[0]
    lines = [
        f"### 推荐 **{winner['label']}**",
        "",
        f"结构得分 {winner['struct_score']:.3f}　｜　{winner['chunks']} 块　｜　"
        f"平均 {winner['avg_chars']} 字",
        "",
        "> 结构代理指标，不是检索精度。块数过少可能只是把整篇塞进两块，"
        "所以同时看平均字数与条款完整率。",
    ]
    return "\n".join(lines)


def _chunk_rows(chunks: list) -> list[list]:
    return [
        [
            chunk.idx,
            f"{chunk.page_from}" if chunk.page_from == chunk.page_to else f"{chunk.page_from}-{chunk.page_to}",
            chunk.kind,
            chunk.section or "—",
            chunk.chars,
            chunk_preview(chunk),
        ]
        for chunk in chunks
    ]


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
                item["page"],
                {"table": "表格", "figure": "图注", "ocr": "OCR"}.get(item["kind"], "正文"),
                fmt("fused"),
                fmt("dense"),
                fmt("sparse"),
                fmt("rerank"),
                " ".join(item["text"].split())[:110],
            ]
        )
    return rows


def _timing_markdown(timings: dict, rerank: dict, degraded: str, query: str, rewritten: bool) -> str:
    parts = [
        f"- 检索查询：`{query}`" + ("（已按上一轮改写）" if rewritten else ""),
        "- 耗时：" + "　".join(
            f"{label} {int(timings.get(key, 0))}ms"
            for key, label in (
                ("rewrite_ms", "改写"),
                ("embed_ms", "向量化"),
                ("retrieve_ms", "检索合计"),
                ("rerank_ms", "精排"),
                ("generate_ms", "生成"),
                ("ttft_ms", "首字"),
                ("total_ms", "合计"),
            )
            if timings.get(key)
        ),
    ]
    if rerank.get("used"):
        parts.append(f"- 精排：`{rerank.get('model')}` 已生效（{int(rerank.get('ms', 0))}ms）")
    elif rerank.get("error"):
        parts.append(f"- 精排未生效：{rerank['error']}")
    if degraded:
        parts.append(f"- 降级：{degraded}")
    return "\n".join(parts)


def build_ui(service: Service) -> gr.Blocks:
    llm_state = f"已配置 · `{settings.llm_model}`" if settings.llm_configured else "**未配置**"
    stats = service.library.stats()

    with gr.Blocks(title=f"rag-agent v{VERSION}") as demo:
        gr.Markdown(
            f"# PDF 助手 v{VERSION}\n"
            f"问答 {llm_state}　｜　向量 `{settings.embed_model}`"
            f"（{'可用' if settings.embed_ready else '未配置'}）　｜　"
            f"精排 `{settings.rerank_model}`（{'可用' if settings.rerank_ready else '未启用'}）　｜　"
            f"索引 {stats['docs']} 篇 / {stats['chunks']} 块\n\n"
            "上传 PDF → 自动切分建索引 → 混合检索 + 交叉编码器精排 → 带编号引用的回答。"
        )

        doc_state = gr.State(None)
        page_state = gr.State([])
        chunk_state = gr.State([])
        source_state = gr.State([])
        history_state = gr.State([])

        # ---------------- 文档 ----------------
        with gr.Tab("文档"):
            with gr.Row():
                with gr.Column(scale=4):
                    upload = gr.File(
                        label="上传 PDF（可多选）",
                        file_types=[".pdf"],
                        file_count="multiple",
                    )
                    auto_index = gr.Checkbox(label="入库后自动建索引", value=True)
                    upload_btn = gr.Button("解析并入库", variant="primary")
                    status = gr.Markdown("等待上传。")
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
                            label="当前文档", choices=_doc_choices(service), value=None, scale=5
                        )
                        delete_btn = gr.Button("删除当前文档", variant="stop", scale=2)
                        reindex_btn = gr.Button("重建索引", scale=2)

            with gr.Row():
                only_suspect = gr.Checkbox(label="仅显示可疑页", value=False, scale=2)
                view_dd = gr.Dropdown(label="查看页码", choices=[], value=None, scale=3)
            page_table = gr.Dataframe(
                headers=PAGE_HEADERS, interactive=False, wrap=True, label="页级统计与路由信号"
            )
            page_box = gr.Textbox(label="单页原文", lines=14, max_lines=30)

            with gr.Accordion("增强识别（扫描页 OCR / 表格结构化 / 图形页图注）", open=False):
                gr.Markdown(
                    f"扫描页 → OCR `{settings.ocr_model}`；表格页 → pdfplumber（离线）；"
                    f"图形页 → 图注 `{settings.vision_model}`。"
                    "**只有「运行增强识别」会调用视觉模型并计费**；增强结果写入数据库，重启后不必重跑。"
                )
                with gr.Row():
                    plan_btn = gr.Button("只看路由计划（不调用模型）", scale=3)
                    enhance_btn = gr.Button("运行增强识别", variant="primary", scale=3)
                    limit_pages = gr.Slider(
                        label="最多调用模型的页数",
                        minimum=1,
                        maximum=60,
                        value=settings.ocr_max_pages,
                        step=1,
                        scale=3,
                    )
                    force_enhance = gr.Checkbox(
                        label="重跑已增强页（重复计费）", value=False, scale=2
                    )
                enhance_summary = gr.Markdown()
                enhance_table = gr.Dataframe(
                    headers=ENHANCE_HEADERS, interactive=False, wrap=True, label="逐页判定与耗时"
                )
                with gr.Row():
                    enhance_dd = gr.Dropdown(label="查看页码", choices=[], value=None, scale=2)
                    enhance_box = gr.Textbox(label="增强后文本", lines=16, max_lines=40, scale=8)

        # ---------------- 切分与索引 ----------------
        with gr.Tab("切分与索引"):
            gr.Markdown(
                "四种策略共用同一份文本，差异只在切分点。**自动推荐**读取文档编号结构，"
                "不调用模型。建索引时若配置了 Embedding Key，会把每个块向量化后写入数据库。"
            )
            with gr.Row():
                profile_btn = gr.Button("分析文档结构", scale=2)
                strategy_dd = gr.Dropdown(
                    label="策略", choices=["auto"] + list(LABELS)[:4], value="auto", scale=3
                )
                size_slider = gr.Slider(
                    label="块长上限（字）", minimum=200, maximum=1200, value=settings.chunk_size, step=50, scale=2
                )
                overlap_slider = gr.Slider(
                    label="重叠（字）", minimum=0, maximum=300, value=settings.chunk_overlap, step=10, scale=2
                )
            with gr.Row():
                chunk_btn = gr.Button("按当前策略切分", variant="primary", scale=2)
                compare_btn = gr.Button("四种策略对比", scale=2)
                index_btn = gr.Button("按当前策略建索引", variant="primary", scale=2)
            profile_box = gr.Markdown()
            metrics_box = gr.Markdown()
            index_box = gr.Markdown()
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
                chunk_dd = gr.Dropdown(label="查看块", choices=[], value=None, scale=2)
                chunk_box = gr.Textbox(label="块全文", lines=14, max_lines=32, scale=8)

        # ---------------- 检索调试 ----------------
        with gr.Tab("检索调试"):
            gr.Markdown(
                "同一问题分别走**向量**、**BM25**、**RRF 融合**，再对比精排前后。"
                "BM25 对编号与缩写（如 `DEMO-001`、`v1.0`）敏感，向量对同义改写更稳。"
            )
            with gr.Row():
                debug_query = gr.Textbox(
                    label="检索问题", placeholder="例如：付款账期最长是多少天？", scale=6
                )
                debug_scope = gr.Dropdown(
                    label="范围（留空=全部文档）",
                    choices=_doc_choices(service),
                    multiselect=True,
                    value=[],
                    scale=4,
                )
                debug_topk = gr.Slider(label="返回条数", minimum=1, maximum=10, value=5, step=1, scale=2)
            debug_btn = gr.Button("对比三种检索", variant="primary")
            debug_summary = gr.Markdown()
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### 向量检索")
                    dense_table = gr.Dataframe(headers=SOURCE_HEADERS[1:], interactive=False, wrap=True)
                with gr.Column():
                    gr.Markdown("#### BM25")
                    sparse_table = gr.Dataframe(headers=SOURCE_HEADERS[1:], interactive=False, wrap=True)
                with gr.Column():
                    gr.Markdown("#### RRF 融合")
                    rrf_table = gr.Dataframe(headers=SOURCE_HEADERS[1:], interactive=False, wrap=True)
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### 精排前（RRF）")
                    before_table = gr.Dataframe(
                        headers=["#", "文件", "页", "类型", "融合分", "预览"], interactive=False, wrap=True
                    )
                with gr.Column():
                    gr.Markdown("#### 精排后（交叉编码器）")
                    after_table = gr.Dataframe(
                        headers=["#", "文件", "页", "类型", "精排分", "预览"], interactive=False, wrap=True
                    )
            debug_box = gr.Textbox(label="命中块全文（从下方选择）", lines=12, max_lines=28)
            debug_dd = gr.Dropdown(label="查看命中块", choices=[], value=None)
            debug_state = gr.State([])

        # ---------------- 智能问答 ----------------
        with gr.Tab("智能问答"):
            gr.Markdown(
                "多轮对话：带指代的问题会先用模型改写为独立问题再检索（界面上会显示改写结果）。"
                "回答逐字流式输出，引用编号可对照下方来源表。"
            )
            with gr.Row():
                chat_scope = gr.Dropdown(
                    label="问答范围（留空=全部文档）",
                    choices=_doc_choices(service),
                    multiselect=True,
                    value=[],
                    scale=4,
                )
                chat_topk = gr.Slider(label="引用块数", minimum=2, maximum=12, value=settings.retrieve_top_k, step=1, scale=2)
                chat_strategy = gr.Radio(
                    label="检索策略", choices=["rrf", "dense", "sparse"], value=settings.retrieve_strategy, scale=2
                )
                chat_rerank = gr.Checkbox(label="启用精排", value=settings.rerank_enabled, scale=1)

            chatbot = gr.Chatbot(label="对话", height=380, show_label=True)
            with gr.Row():
                chat_input = gr.Textbox(
                    label="问题", placeholder="例如：预付款的比例上限和担保要求是什么？", scale=9
                )
                send_btn = gr.Button("发送", variant="primary", scale=1)
                clear_btn = gr.Button("清空对话", scale=1)

            chat_meta = gr.Markdown(label="检索与耗时")
            source_table = gr.Dataframe(
                headers=SOURCE_HEADERS, interactive=False, wrap=True, label="本次引用来源"
            )
            with gr.Row():
                source_dd = gr.Dropdown(label="展开引用原文", choices=[], value=None, scale=2)
                source_box = gr.Textbox(label="引用块原文", lines=12, max_lines=30, scale=8)

        # ---------------- 评测 ----------------
        with gr.Tab("评测"):
            gr.Markdown(
                "`eval/golden.jsonl` 中的问题带标注页码。"
                "**hit@k** 看前 k 条是否命中标注页，**MRR** 看第一个命中的名次；"
                "勾选「含生成指标」后额外统计引用有效率与拒答准确率（每题一次模型调用）。"
            )
            with gr.Row():
                eval_pattern = gr.Textbox(label="文档名包含（留空=全部）", value="", scale=3)
                eval_topk = gr.Slider(label="k", minimum=1, maximum=10, value=5, step=1, scale=2)
                eval_strategy = gr.Radio(label="检索策略", choices=["rrf", "dense", "sparse"], value="rrf", scale=3)
                eval_generate = gr.Checkbox(label="含生成指标（计费）", value=False, scale=2)
                eval_limit = gr.Slider(label="最多题数（0=全部）", minimum=0, maximum=40, value=0, step=1, scale=2)
            with gr.Row():
                eval_btn = gr.Button("运行评测", variant="primary", scale=2)
                export_btn = gr.Button("导出 Markdown 到 eval/report.md", scale=3)
            eval_summary = gr.Markdown()
            eval_table = gr.Dataframe(headers=EVAL_HEADERS, interactive=False, wrap=True, label="逐题结果")
            eval_state = gr.State([])

        # ---------------- 系统信息 ----------------
        with gr.Tab("系统信息"):
            with gr.Row():
                sys_btn = gr.Button("刷新", variant="primary", scale=1)
            sys_md = gr.Markdown(_system_markdown(service))
            query_table = gr.Dataframe(
                headers=["时间", "问题", "改写", "来源页", "策略", "精排", "耗时ms"],
                value=_query_rows(service),
                interactive=False,
                wrap=True,
                label="最近查询",
            )

        # ================= handlers =================

        def on_upload(files, also_index, progress=gr.Progress()):
            paths = _file_paths(files)
            if not paths:
                return (
                    "请先选择 PDF 文件。", _doc_rows(service),
                    _update(choices=_doc_choices(service), value=None), None,
                    _update(choices=_doc_choices(service), value=[]),
                    _update(choices=_doc_choices(service), value=[]),
                )

            lines: list[str] = []
            last_id = None
            for index, path in enumerate(paths, start=1):
                if not path.lower().endswith(".pdf"):
                    lines.append(f"- 跳过非 PDF：`{Path(path).name}`")
                    continue
                progress((index - 1) / max(len(paths), 1), desc=f"解析 {Path(path).name}")
                started = time.perf_counter()
                try:
                    doc, existed = service.ingest(path, Path(path).name)
                except ValueError as exc:
                    lines.append(f"- **{Path(path).name}** 解析失败：{exc}")
                    continue
                elapsed = int((time.perf_counter() - started) * 1000)
                lines.append(
                    f"- {'已存在' if existed else '已入库'} **{doc['filename']}**："
                    f"{doc['page_count']} 页 / {doc['chars']:,} 字符（{elapsed} ms）"
                )
                last_id = doc["id"]

                if also_index:
                    try:
                        info = service.index(doc["id"], force=True)
                        note = (
                            f"　→　索引 {info['chunks']} 块 / {info['dim']} 维"
                            f"（{info['strategy']}，{info['embed_ms']:.0f} ms）"
                        )
                        if info["embed_error"]:
                            note += f"\n  - Embedding 失败：{info['embed_error']}（已退化为纯 BM25 索引）"
                        lines.append(note)
                    except ServiceError as exc:
                        lines.append(f"  - 建索引失败：{exc}")

            value = last_id
            return (
                "### 入库结果\n\n" + "\n".join(lines),
                _doc_rows(service),
                _update(choices=_doc_choices(service), value=value),
                value,
                _update(choices=_doc_choices(service), value=[]),
                _update(choices=_doc_choices(service), value=[]),
            )

        def on_doc_change(doc_id):
            doc = service.docs.get(doc_id) if doc_id else None
            if doc is None:
                return (
                    "请选择或上传一份 PDF。",
                    [],
                    [],
                    _update(choices=[], value=None),
                    _update(choices=[], value=None),
                    _update(choices=[], value=None),
                )
            rows, pages = _page_rows(doc)
            blocks = doc.get("blocks") or {}
            enhanced_pages = sorted(blocks) or pages
            chunk_choices = [
                (f"#{chunk.idx} · {chunk.chars} 字 · {chunk.kind}", chunk.idx)
                for chunk in service.library.records
                if chunk.doc_id == doc["id"]
            ]
            return (
                _doc_summary(doc),
                rows,
                pages,
                _update(choices=pages, value=None),
                _update(choices=enhanced_pages, value=None),
                _update(choices=chunk_choices, value=None),
                _update(choices=pages, value=None),
            )

        def on_filter(show_suspect, doc_id):
            doc = service.docs.get(doc_id) if doc_id else None
            if doc is None:
                return [], []
            return _page_rows(doc, bool(show_suspect))

        def on_page_view(page_no, doc_id):
            doc = service.docs.get(doc_id) if doc_id else None
            if doc is None:
                return "请先选择文档。"
            if not page_no:
                return "请选择页码。"
            return page_text_view(doc, int(page_no))

        def on_page_select(evt: gr.SelectData, pages, doc_id):
            raw = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
            if not pages or not isinstance(raw, int) or raw >= len(pages):
                return "未选中有效行。"
            return on_page_view(pages[raw], doc_id)

        def on_delete(doc_id):
            choices = _doc_choices(service)
            if not doc_id:
                return (
                    "请先选择文档。", _doc_rows(service),
                    _update(choices=choices, value=None), None,
                    _update(choices=choices, value=[]),
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
                _update(choices=choices, value=[]),
                _update(choices=choices, value=[]),
            )

        def on_reindex(doc_id, strategy, size, overlap):
            if not doc_id:
                return "请先选择文档。", _doc_rows(service)
            try:
                info = service.index(
                    doc_id, strategy=strategy, size=int(size), overlap=int(overlap), force=True
                )
            except ServiceError as exc:
                return f"建索引失败：{exc}", _doc_rows(service)
            note = (
                f"### 索引已重建\n\n块数 **{info['chunks']}**　｜　维度 **{info['dim']}**　｜　"
                f"策略 `{info['strategy']}`　｜　向量化 {info['embed_ms']:.0f} ms　｜　总耗时 {info['ms']:.0f} ms"
            )
            if info["embed_error"]:
                note += f"\n\n> Embedding 未生效：{info['embed_error']}\n> 当前索引只能走 BM25 关键词检索。"
            return note, _doc_rows(service)

        def on_plan(doc_id):
            doc = service.docs.get(doc_id) if doc_id else None
            if doc is None:
                return "请先选择文档。", [], _update(choices=[], value=None)
            plans = plan_pages(doc)
            rows = [
                [p.page, " + ".join(p.routes), p.reason[:60], "计划中", p.chars, "", "", "—", ""]
                for p in plans
            ]
            counts: dict[str, int] = {}
            for plan in plans:
                for route in plan.routes:
                    counts[route] = counts.get(route, 0) + 1
            labels = {"text": "文本层", "table": "表格", "ocr": "OCR", "figure": "图注"}
            breakdown = "　｜　".join(f"{labels.get(k, k)} {v} 页" for k, v in sorted(counts.items()))
            need = sum(1 for plan in plans if "ocr" in plan.routes or "figure" in plan.routes)
            return (
                f"### 路由计划（未调用任何模型）\n\n{breakdown}\n\n"
                f"- 需要视觉模型的页：{need} 页（上限 `OCR_MAX_PAGES={settings.ocr_max_pages}`）",
                rows,
                _update(choices=[p.page for p in plans], value=None),
            )

        def on_enhance(doc_id, limit, force, progress=gr.Progress()):
            doc = service.docs.get(doc_id) if doc_id else None
            if doc is None:
                return "请先选择文档。", [], _update(choices=[], value=None)
            if not settings.ocr_enabled:
                return (
                    "OCR 未启用：在 `.env` 设置 `OCR_ENABLED=true` 后重启。"
                    "表格通道不依赖 OCR。",
                    [],
                    _update(choices=[], value=None),
                )

            done = {"n": 0}
            total = max(min(int(limit), doc["page_count"]), 1)

            def on_event(event):
                if event.get("stage") == "page_start":
                    done["n"] += 1
                    progress(
                        min(done["n"] / total, 1.0),
                        desc=f"第 {event['page']} 页 → {'+'.join(event.get('routes', []))}",
                    )

            report = service.enhance(
                doc_id, limit_pages=int(limit), on_event=on_event, force=bool(force)
            )
            reindexed = ""
            if report["api_calls"] and doc.get("indexed_at"):
                try:
                    info = service.index(doc_id, force=True)
                    reindexed = (
                        f"\n- 已自动重建索引：{info['chunks']} 块 / {info['dim']} 维"
                        f"（{info['embed_ms']:.0f} ms）"
                    )
                except ServiceError as exc:
                    reindexed = f"\n- 自动重建索引失败：{exc}"
            rows = []
            for result in report["pages"]:
                plan = next((p for p in report["plans"] if p.page == result.page), None)
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
            summary = (
                "### 增强识别完成（结果已入库，重启后保留）\n\n"
                f"- 调用视觉模型 **{report['api_calls']}** 次"
                + (f"，复用已增强 {report['cached']} 页" if report.get("cached") else "")
                + "\n"
                f"- 文本量 {report['text_before']:,} → {report['text_after']:,} 字符\n"
                f"- 总耗时 {report['ms'] / 1000:.1f} s"
                + (f"\n- 因预算跳过：{format_pages(report['skipped'])}" if report["skipped"] else "")
                + (reindexed or "\n- 切分/索引结果来自增强前的文本，重建索引后才会用上增强内容。")
            )
            pages = [r.page for r in report["pages"]]
            return summary, rows, _update(choices=pages, value=None)

        def on_enhance_view(page_no, doc_id):
            doc = service.docs.get(doc_id) if doc_id else None
            if doc is None:
                return "请先选择文档。"
            if not page_no:
                return "请选择页码。"
            blocks = blocks_of(doc, int(page_no))
            if not blocks:
                return f"第 {page_no} 页没有可显示的文本。"
            parts = []
            for block in blocks:
                label = {"text": "原文", "ocr": "OCR", "table": "表格", "figure": "图注"}.get(block.type, block.type)
                parts.append(f"[{label} · {block.source} · {len(block.text)} 字]\n{block.text}")
            return "\n\n" + ("\n\n" + "-" * 48 + "\n\n").join(parts)

        def on_profile(doc_id):
            doc = service.docs.get(doc_id) if doc_id else None
            if doc is None:
                return "请先选择文档。"
            profile = profile_document(
                all_blocks(doc),
                pages=doc["page_count"],
                table_pages=len(doc["table_like_pages"]),
                figure_pages=len(doc["figure_like_pages"]),
            )
            return _profile_markdown(profile)

        def on_chunk(doc_id, strategy, size, overlap):
            doc = service.docs.get(doc_id) if doc_id else None
            if doc is None:
                return "请先选择文档。", [], [], _update(choices=[], value=None), _update(value=[])
            result = chunk_document(
                all_blocks(doc),
                strategy=strategy,
                size=int(size),
                overlap=int(overlap),
                min_chars=settings.chunk_min_chars,
                percentile=settings.semantic_percentile,
                pages=doc["page_count"],
                table_pages=len(doc["table_like_pages"]),
                figure_pages=len(doc["figure_like_pages"]),
            )
            chunks = result["chunks"]
            choices = [(f"#{c.idx} · {c.chars} 字 · {c.kind}", c.idx) for c in chunks]
            return (
                _metrics_markdown(result),
                _chunk_rows(chunks),
                chunks,
                _update(choices=choices, value=None),
                [],
            )

        def on_compare(doc_id, size, overlap):
            doc = service.docs.get(doc_id) if doc_id else None
            if doc is None:
                return "请先选择文档。", []
            rows = compare_strategies(
                all_blocks(doc),
                size=int(size),
                overlap=int(overlap),
                min_chars=settings.chunk_min_chars,
                percentile=settings.semantic_percentile,
                pages=doc["page_count"],
                table_pages=len(doc["table_like_pages"]),
                figure_pages=len(doc["figure_like_pages"]),
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
                        f"{chunk.kind} · {chunk.chars} 字]\n章节：{chunk.section or '—'}\n\n{chunk.text}"
                    )
            return "未找到该块。"

        def on_debug(query, scope, top_k):
            if not query or not query.strip():
                return "请输入检索问题。", [], [], [], [], [], [], [], []
            try:
                result = service.debug(query, scope or None, int(top_k))
            except ServiceError as exc:
                return f"检索失败：{exc}", [], [], [], [], [], [], [], []

            def rail_rows(key: str) -> list[list]:
                return [
                    [
                        item["filename"],
                        item["page"],
                        {"table": "表格", "figure": "图注", "ocr": "OCR"}.get(item["kind"], "正文"),
                        f"{item['scores'][key]:.3f}",
                        item["preview"],
                    ]
                    for item in result["rails"][key]
                ]

            before = [
                [item["n"], item["filename"], item["page"], item["kind"], f"{item['scores']['fused']:.4f}", item["preview"]]
                for item in result["rails"]["rrf"]
            ]
            after = [
                [
                    item["n"],
                    item["filename"],
                    item["page"],
                    item["kind"],
                    "—" if item["scores"]["rerank"] is None else f"{item['scores']['rerank']:.3f}",
                    item["preview"],
                ]
                for item in result["reranked"]
            ]
            rerank_note = (
                f"精排 `{result['rerank'].get('model')}` 生效（{result['rerank']['ms']:.0f}ms）"
                if result["rerank"]["used"]
                else f"精排未生效：{result['rerank'].get('error') or '已关闭'}"
            )
            summary = (
                f"查询 `{query}`　｜　耗时 "
                + "　".join(f"{k} {v:.0f}ms" for k, v in result["timings"].items())
                + f"\n\n- {rerank_note}"
                + (f"\n- 降级：{result['degraded']}" if result["degraded"] else "")
            )
            combined = result["rails"]["rrf"] + result["reranked"]
            choices = [
                (f"#{item['n']} · {item['filename']} p.{item['page']}", position)
                for position, item in enumerate(combined)
            ]
            return (
                summary,
                rail_rows("dense"),
                rail_rows("sparse"),
                rail_rows("rrf"),
                before,
                after,
                combined,
                _update(choices=choices, value=None),
                "",
            )

        def on_debug_view(position, items):
            if not items:
                return "请先运行对比。"
            if position is None:
                return "请选择一个命中块。"
            try:
                item = items[int(position)]
            except (IndexError, TypeError, ValueError):
                return "未找到该块。"
            scores = item["scores"]
            vector = "—" if scores["dense"] is None else f"{scores['dense']:.3f}"
            sparse = "—" if scores["sparse"] is None else f"{scores['sparse']:.2f}"
            rerank = "—" if scores["rerank"] is None else f"{scores['rerank']:.3f}"
            return (
                f"[{item['n']}] 《{item['filename']}》p.{item['page']} · {item['kind']} · {item['chars']} 字\n"
                f"融合 {scores['fused']:.4f}　向量 {vector}　BM25 {sparse}　精排 {rerank}\n"
                f"章节：{item.get('section') or '—'}\n\n{item['text']}"
            )

        def _source_choices(sources: list[dict]) -> list[tuple[str, int]]:
            return [(f"[{item['n']}] 《{item['filename']}》p.{item['page']}", item["n"]) for item in sources]

        def on_chat(message, history, scope, top_k, strategy, do_rerank):
            history = list(history or [])
            if not message or not message.strip():
                yield history, "", "", [], history, [], _update(choices=[], value=None)
                return
            if not service.library.records:
                note = "索引为空：请先在「文档」页上传 PDF 并建索引。"
                history = history + [
                    {"role": "user", "content": message},
                    {"role": "assistant", "content": note},
                ]
                yield history, "", "", [], history, [], _update(choices=[], value=None)
                return

            past = history[-settings.history_turns * 2 :]
            history = history + [{"role": "user", "content": message}]
            answer = ""
            meta = "检索中…"
            sources: list[dict] = []
            for event in service.ask_stream(
                message,
                doc_ids=scope or None,
                top_k=int(top_k),
                strategy=strategy,
                rerank=do_rerank,
                history=past,
            ):
                stage = event["stage"]
                if stage == "prepared":
                    sources = event["sources"]
                    meta = _timing_markdown(
                        event["timings"], event["rerank"], event["degraded"],
                        event["query"], event["rewritten"],
                    )
                    yield (
                        history + [{"role": "assistant", "content": "…"}],
                        "",
                        meta,
                        _source_rows(sources),
                        history,
                        sources,
                        _update(choices=_source_choices(sources), value=None),
                    )
                elif stage == "delta":
                    answer += event["text"]
                    yield (
                        history + [{"role": "assistant", "content": answer}],
                        "",
                        meta,
                        _source_rows(sources),
                        history,
                        sources,
                        _update(choices=_source_choices(sources), value=None),
                    )
                elif stage == "error":
                    history = history + [
                        {"role": "assistant", "content": f"调用模型失败：{event['error']}"}
                    ]
                    yield (
                        history, "", meta, _source_rows(sources), history, sources,
                        _update(choices=_source_choices(sources), value=None),
                    )
                    return
                elif stage == "done":
                    result = event["result"]
                    sources = result["sources"]
                    meta = _timing_markdown(
                        result["timings"], result["rerank"], result["degraded"],
                        result["query"], result["rewritten"],
                    )
                    meta += (
                        f"\n- 引用：{result['citations']['valid']}/{result['citations']['total']} 有效"
                        if result["citations"]["total"]
                        else "\n- 本次回答未标注引用编号"
                    )
                    if result["abstained"]:
                        meta += "\n- 判定为拒答（资料不足）"
                    history = history + [{"role": "assistant", "content": result["answer"]}]
                    yield (
                        history, "", meta, _source_rows(sources), history, sources,
                        _update(choices=_source_choices(sources), value=None),
                    )
                    return

        def on_clear():
            return [], "", "", [], [], [], _update(choices=[], value=None)

        def on_source_view(number, sources):
            if not sources:
                return "本次没有引用来源。"
            if number is None:
                return "请选择一条引用。"
            for item in sources:
                if item["n"] == int(number):
                    scores = item["scores"]
                    vector = "—" if scores["dense"] is None else f"{scores['dense']:.3f}"
                    return (
                        f"[{item['n']}] 《{item['filename']}》p.{item['page']} · {item['kind']} · {item['chars']} 字\n"
                        f"融合 {scores['fused']:.4f}　向量 {vector}\n"
                        f"章节：{item.get('section') or '—'}\n\n{item['text']}"
                    )
            return "未找到该引用。"

        def on_eval(pattern, top_k, strategy, with_gen, limit):
            try:
                result = service.evaluate(
                    doc_pattern=pattern,
                    top_k=int(top_k),
                    strategy=strategy,
                    with_generation=bool(with_gen),
                    limit=int(limit) or None,
                )
            except Exception as exc:
                return f"评测失败：{exc}", [], []
            summary = result["summary"]
            if not summary["questions"]:
                return (
                    f"没有可评测的问题：`eval/golden.jsonl` 中的文档需要在当前索引里"
                    f"（本次跳过 {result['skipped']} 条）。",
                    [],
                    [],
                )
            lines = [
                f"### hit@{summary['k']} **{summary['hit']:.1%}**　｜　MRR **{summary['mrr']:.3f}**",
                "",
                f"- 问题 {summary['questions']} 条（answerable {summary['answerable']}），"
                f"跳过 {result['skipped']} 条",
            ]
            if summary["grounding"] is not None:
                lines.append(f"- 引用有效率：{summary['grounding']:.1%}（{summary['citation_total']} 处引用）")
            if summary["abstain_accuracy"] is not None:
                lines.append(f"- 拒答准确率：{summary['abstain_accuracy']:.1%}")
            rows = [
                [
                    row["id"],
                    row["q"][:44],
                    str(row["pages"][:5]),
                    str(row["gold"]),
                    "✓" if row["hit"] else "✗",
                    f"{row['rr']:.2f}",
                    row["ms"],
                ]
                for row in result["rows"]
            ]
            return "\n".join(lines), rows, result["rows"]

        def on_export(rows):
            if not rows:
                return "请先运行一次评测。"
            from eval.run_eval import format_markdown
            from eval.metrics import summarize

            summary = summarize(rows, 5)
            markdown = format_markdown(
                rows, summary, 0, {"strategy": settings.retrieve_strategy, "model": settings.llm_model,
                                   "chunks": service.library.stats()["chunks"]}
            )
            target = BASE_DIR / "eval" / "report.md"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(markdown, encoding="utf-8")
            return f"已写入 `{target}`（{len(markdown)} 字符）。"

        # ================= bindings =================

        upload_btn.click(
            fn=on_upload,
            inputs=[upload, auto_index],
            outputs=[status, doc_table, doc_dd, doc_state, debug_scope, chat_scope],
            show_progress=True,
        )
        doc_dd.change(
            fn=on_doc_change,
            inputs=[doc_dd],
            outputs=[status, page_table, page_state, view_dd, enhance_dd, chunk_dd],
            show_progress=True,
        )
        only_suspect.change(fn=on_filter, inputs=[only_suspect, doc_state], outputs=[page_table, page_state])
        page_table.select(
            fn=on_page_select,
            inputs=[page_state, doc_state],
            outputs=page_box,
        )
        view_dd.change(fn=on_page_view, inputs=[view_dd, doc_state], outputs=page_box)
        delete_btn.click(
            fn=on_delete,
            inputs=[doc_state],
            outputs=[status, doc_table, doc_dd, doc_state, debug_scope, chat_scope],
        )
        reindex_btn.click(
            fn=on_reindex,
            inputs=[doc_state, strategy_dd, size_slider, overlap_slider],
            outputs=[index_box, doc_table],
            show_progress=True,
        )

        plan_btn.click(
            fn=on_plan,
            inputs=[doc_state],
            outputs=[enhance_summary, enhance_table, enhance_dd],
            show_progress=True,
        )
        enhance_btn.click(
            fn=on_enhance,
            inputs=[doc_state, limit_pages, force_enhance],
            outputs=[enhance_summary, enhance_table, enhance_dd],
            show_progress=True,
        )
        enhance_dd.change(fn=on_enhance_view, inputs=[enhance_dd, doc_state], outputs=enhance_box)

        profile_btn.click(fn=on_profile, inputs=[doc_state], outputs=profile_box, show_progress=True)
        chunk_btn.click(
            fn=on_chunk,
            inputs=[doc_state, strategy_dd, size_slider, overlap_slider],
            outputs=[metrics_box, chunk_table, chunk_state, chunk_dd, compare_table],
            show_progress=True,
        )
        compare_btn.click(
            fn=on_compare,
            inputs=[doc_state, size_slider, overlap_slider],
            outputs=[metrics_box, compare_table],
            show_progress=True,
        )
        chunk_dd.change(fn=on_chunk_view, inputs=[chunk_dd, chunk_state], outputs=chunk_box)
        index_btn.click(
            fn=on_reindex,
            inputs=[doc_state, strategy_dd, size_slider, overlap_slider],
            outputs=[index_box, doc_table],
            show_progress=True,
        )

        debug_btn.click(
            fn=on_debug,
            inputs=[debug_query, debug_scope, debug_topk],
            outputs=[
                debug_summary, dense_table, sparse_table, rrf_table,
                before_table, after_table, debug_state, debug_dd, debug_box,
            ],
            show_progress=True,
        )
        debug_dd.change(fn=on_debug_view, inputs=[debug_dd, debug_state], outputs=debug_box)

        _chat_outputs = [
            chatbot, chat_input, chat_meta, source_table, history_state, source_state, source_dd,
        ]
        send_btn.click(
            fn=on_chat,
            inputs=[chat_input, history_state, chat_scope, chat_topk, chat_strategy, chat_rerank],
            outputs=_chat_outputs,
            show_progress=True,
        )
        chat_input.submit(
            fn=on_chat,
            inputs=[chat_input, history_state, chat_scope, chat_topk, chat_strategy, chat_rerank],
            outputs=_chat_outputs,
            show_progress=True,
        )
        clear_btn.click(fn=on_clear, inputs=[], outputs=_chat_outputs)
        source_dd.change(fn=on_source_view, inputs=[source_dd, source_state], outputs=source_box)

        eval_btn.click(
            fn=on_eval,
            inputs=[eval_pattern, eval_topk, eval_strategy, eval_generate, eval_limit],
            outputs=[eval_summary, eval_table, eval_state],
            show_progress=True,
        )
        export_btn.click(fn=on_export, inputs=[eval_state], outputs=[eval_summary])

        sys_btn.click(
            fn=lambda: (_system_markdown(service), _query_rows(service)),
            inputs=None,
            outputs=[sys_md, query_table],
        )

    return demo


def _system_markdown(service: Service) -> str:
    info = service.system_info()
    index = info["index"]
    security = info["security"]
    return (
        f"### 版本与模型\n\n"
        f"| 项 | 值 |\n| --- | --- |\n"
        f"| 版本 | `{info['version']}` |\n"
        f"| 启动时间 | {info['started_at']} |\n"
        f"| 对话模型 | `{info['llm']['model']}`（{'已配置' if info['llm']['configured'] else '未配置'}） |\n"
        f"| Embedding | `{info['embedding']['model']}`（{'可用' if info['embedding']['configured'] else '未配置'}） |\n"
        f"| 精排 | `{info['rerank']['model']}`（{'可用' if info['rerank']['configured'] else '未启用'}） |\n"
        f"| OCR | `{info['ocr']['model']}`（{'启用' if info['ocr']['enabled'] else '未启用'}） |\n\n"
        f"### 索引规模\n\n"
        f"| 项 | 值 |\n| --- | --- |\n"
        f"| 文档 / 块 / 向量 | {index['docs']} / {index['chunks']} / {index['vectors']} |\n"
        f"| 向量维度 / 词表 | {index['dim']} / {index['vocab']} |\n"
        f"| 数据库 | {info['db']['db_bytes'] / 1024:.0f} KB，累计提问 {info['db']['queries']} 次 |\n\n"
        f"### 检索与切分配置（只读）\n\n"
        f"| 项 | 值 |\n| --- | --- |\n"
        f"| 默认策略 / top_k / 候选 / RRF k | {info['retrieval']['strategy']} / "
        f"{info['retrieval']['top_k']} / {info['retrieval']['candidates']} / {info['retrieval']['rrf_k']} |\n"
        f"| 切分 | {info['chunking']['strategy']} · {info['chunking']['size']} 字 · "
        f"重叠 {info['chunking']['overlap']} |\n"
        f"| 访问控制 | API Key {'必需' if security['api_key_required'] else '未启用'}"
        f"　｜　界面登录 {'启用' if security['ui_login'] else '未启用'}"
        f"　｜　限流 {security['rate_limit_per_min']}/分钟 |\n"
    )


def _query_rows(service: Service) -> list[list]:
    rows = []
    for item in service.db.recent_queries(15):
        rows.append(
            [
                item["created_at"],
                item["question"][:40],
                item["rewritten"][:30],
                item["pages"],
                item["strategy"],
                "是" if item["reranked"] else "",
                item["ms"],
            ]
        )
    return rows
