"""UI smoke test. Nothing else imports app/ui, so a syntax or component error
would only surface at server start."""

import pytest

pytest.importorskip("gradio")
pytest.importorskip("pydantic_settings")

from app.ui import _metrics_markdown, _source_rows, build_ui  # noqa: E402
from rag.service import Service  # noqa: E402
from rag.store.db import Database  # noqa: E402

TABS = ("文档", "切分与索引", "检索调试", "智能问答", "评测", "系统信息")


@pytest.fixture
def demo(tmp_path):
    return build_ui(Service(Database(tmp_path / "ui.sqlite")))


def test_ui_builds(demo):
    assert demo is not None
    assert len(demo.blocks) > 60


def test_all_tabs_exist(demo):
    labels = [
        getattr(block, "label", None)
        for block in demo.blocks.values()
        if isinstance(getattr(block, "label", None), str)
    ]
    for expected in TABS:
        assert expected in labels, f"缺少标签页 {expected}"


def test_metrics_markdown_handles_missing_terms():
    """A document with no clauses and no sentences must not print fake 100%."""
    result = {
        "strategy": "structured",
        "ms": 3,
        "info": {},
        "metrics": {
            "chunks": 4,
            "avg_chars": 300,
            "chunks_per_1k": 2.0,
            "clause_integrity": 1.0,
            "clause_applicable": False,
            "boundary_violation": 0.9,
            "boundary_applicable": False,
            "prose_ratio": 0.05,
            "size_conformance": 0.9,
            "section_coverage": 0.0,
            "struct_score": 0.8,
        },
    }
    text = _metrics_markdown(result)
    assert "不适用" in text
    assert "struct_score" not in text


def test_source_rows_tolerate_missing_scores():
    rows = _source_rows(
        [
            {
                "n": 1,
                "filename": "a.pdf",
                "page": 3,
                "kind": "table",
                "text": "| a | b |",
                "scores": {"fused": 0.03, "dense": None, "sparse": 12.5, "rerank": None},
            }
        ]
    )
    assert rows[0][0] == 1
    assert rows[0][4] == "0.030"
    assert rows[0][5] == "—"
