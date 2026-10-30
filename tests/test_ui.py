"""UI smoke test. Nothing else imports app/ui, so a syntax or component error
would only surface at server start."""

import pytest

pytest.importorskip("gradio")
pytest.importorskip("pydantic_settings")

from app.ui import (  # noqa: E402
    PLAN_STEPS,
    STEPS,
    _source_rows,
    build_ui,
    chunk_map_html,
    chunk_summary_md,
    compare_markdown,
    flow_html,
    plan_steps,
    render_trace,
    step_states,
)
from rag.service import Service  # noqa: E402
from rag.store.db import Database  # noqa: E402

TABS = ("① 文档工作台", "② 提问")


@pytest.fixture
def demo(tmp_path):
    return build_ui(Service(Database(tmp_path / "ui.sqlite")))


def test_ui_builds(demo):
    assert demo is not None
    assert len(demo.blocks) > 40


def test_both_tabs_exist(demo):
    labels = [
        getattr(block, "label", None)
        for block in demo.blocks.values()
        if isinstance(getattr(block, "label", None), str)
    ]
    for expected in TABS:
        assert expected in labels, f"缺少标签页 {expected}"


def test_source_rows_tolerate_missing_scores():
    rows = _source_rows(
        [
            {
                "n": 1,
                "filename": "a.pdf",
                "page": 3,
                "page_to": 3,
                "kind": "table",
                "text": "| a | b |",
                "scores": {"fused": 0.03, "dense": None, "sparse": 12.5, "rerank": None},
            }
        ]
    )
    assert rows[0][0] == 1
    assert rows[0][1] == "a.pdf"
    assert rows[0][4] == "0.030"
    assert rows[0][5] == "—"


class _Chunk:
    def __init__(self, idx, page_from, page_to, chars, kind="text", text="x"):
        self.idx = idx
        self.doc_id = "d1"
        self.page_from = page_from
        self.page_to = page_to
        self.chars = chars
        self.kind = kind
        self.text = text


def test_chunk_map_renders_bars_and_hits():
    chunks = [_Chunk(0, 1, 1, 100), _Chunk(1, 2, 3, 400, "table")]
    page = chunk_map_html(chunks, [{"n": 1, "doc_id": "d1", "page": 2, "page_to": 3}])
    assert "p.2-3" in page
    assert "引用 1" in page
    assert "#0" in page


def test_chunk_map_is_safe_with_no_chunks():
    assert "还没有分块" in chunk_map_html([])


def test_chunk_map_escapes_markup():
    page = chunk_map_html([_Chunk(0, 1, 1, 10, text="<script>alert(1)</script>")])
    assert "<script>alert" not in page
    assert "&lt;script&gt;" in page


def test_chunk_summary_counts_kinds():
    text = chunk_summary_md([_Chunk(0, 1, 1, 100), _Chunk(1, 2, 2, 300, "table")])
    assert "2" in text and "表格 1" in text


def test_trace_shows_running_and_done_states():
    page = render_trace(
        [
            {"title": "向量化查询", "status": "done", "meta": "12 ms", "detail": "1024 维"},
            {"title": "生成回答", "status": "running"},
        ]
    )
    assert "向量化查询" in page
    assert "1024 维" in page
    assert "rt-spin" in page  # the spinner keyframe is present for running steps


def test_trace_placeholder_names_the_steps_of_that_mode():
    fast = render_trace([], "fast")
    enhanced = render_trace([], "enhanced")
    assert "并行检索" in fast and "理解问题" not in fast
    assert "理解问题" in enhanced and "精排" in enhanced


def test_six_step_directory_is_rendered():
    labels = [label for _, label, _ in STEPS]
    assert len(STEPS) == 6
    assert labels[0].endswith("上传建索引") and labels[-1].endswith("检索指标")
    page = flow_html(step_states(None, []))
    for label in labels:
        assert label.replace("① ", "").replace("② ", "").replace("③ ", "").replace("④ ", "").replace("⑤ ", "").replace("⑥ ", "") in page


def test_step_states_mark_what_is_done():
    doc = {"filename": "a.pdf", "format": "PDF", "page_count": 8, "chars": 1967,
           "scan_like_pages": [], "table_like_pages": [4]}
    states = step_states(doc, [_Chunk(0, 1, 1, 100)])
    by_key = {item["key"]: item["done"] for item in states}
    assert by_key["upload"] and by_key["parse"] and by_key["chunks"]
    assert not by_key["metrics"]          # 评测要跑过才算完成
    assert "2 页" not in states[1]["hint"] and "8 页" in states[1]["hint"]


def test_fast_mode_plan_is_shorter_than_enhanced():
    assert PLAN_STEPS["fast"] == ["embed", "retrieve", "generate"]
    assert len(PLAN_STEPS["enhanced"]) > len(PLAN_STEPS["fast"])
    assert [item["status"] for item in plan_steps("enhanced")] == ["pending"] * 7


def test_compare_markdown_explains_both_modes():
    text = compare_markdown(
        {
            "query": "付款账期",
            "rows": [
                {"label": "快速检索", "hits": 5, "pages": 4, "coverage": 0.5, "margin": 0.02,
                 "rerank_used": False, "ms": 12},
                {"label": "增强检索", "hits": 8, "pages": 6, "coverage": 0.75, "margin": 0.05,
                 "rerank_used": True, "ms": 40},
            ],
            "advice": "增强多带回 2 页",
        }
    )
    assert "快速检索" in text and "增强检索" in text
    assert "+25%" in text
    assert "增强多带回 2 页" in text
