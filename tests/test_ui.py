"""UI smoke test. The interface is not imported by other tests, so a syntax or
component error in app/ui.py would only surface at server start."""

import pytest

pytest.importorskip("gradio")
pytest.importorskip("pydantic_settings")

from app.state import STORE  # noqa: E402
from app.ui import build_ui  # noqa: E402


def test_ui_builds():
    demo = build_ui(STORE)
    assert demo is not None
    assert len(demo.blocks) > 20


def test_all_four_tabs_exist():
    demo = build_ui(STORE)
    labels = []
    for block in demo.blocks.values():
        label = getattr(block, "label", None)
        if isinstance(label, str):
            labels.append(label)
    for expected in ("文档", "增强识别", "切分策略", "问答"):
        assert expected in labels, f"缺少标签页 {expected}"


def test_metrics_markdown_handles_missing_terms():
    """A document with no clauses and no sentences must not print fake 100%."""
    from app.ui import _metrics_markdown as metrics_markdown

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
    text = metrics_markdown(result)
    assert "不适用" in text
    assert "struct_score" not in text
