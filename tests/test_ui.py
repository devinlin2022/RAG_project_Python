"""UI smoke test. Nothing else imports app/ui, so a syntax or component error
would only surface at server start."""

import pytest

pytest.importorskip("gradio")
pytest.importorskip("pydantic_settings")

from app.ui import (  # noqa: E402
    _source_rows,
    build_ui,
    chunk_map_html,
    chunk_summary_md,
    render_trace,
)
from rag.service import Service  # noqa: E402
from rag.store.db import Database  # noqa: E402

TABS = ("① 文档与分块", "② 提问")


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


def test_trace_placeholder_mentions_the_pipeline():
    assert "改写" in render_trace([])
