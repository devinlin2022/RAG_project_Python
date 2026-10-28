"""Evaluation metrics and the golden set itself."""

from eval.metrics import (
    citation_grounding,
    cited_numbers,
    hit_at_k,
    is_abstain,
    reciprocal_rank,
    summarize,
)
from eval.run_eval import load_golden


def test_hit_at_k_only_looks_at_k():
    assert hit_at_k([9, 8, 3], [3], 3) == 1.0
    assert hit_at_k([9, 8, 3], [3], 2) == 0.0
    assert hit_at_k([1], [], 5) == 0.0


def test_reciprocal_rank_uses_first_hit():
    assert reciprocal_rank([7, 3, 5], [3]) == 0.5
    assert reciprocal_rank([], [3]) == 0.0


def test_citations_are_deduplicated_in_order():
    assert cited_numbers("结论 [2]，依据 [1][2]") == [2, 1]


def test_out_of_range_citation_is_not_grounded():
    valid, total = citation_grounding("见 [1] 与 [9]", 3)
    assert (valid, total) == (1, 2)


def test_abstain_detection():
    assert is_abstain("资料中未提及该问题。")
    assert is_abstain("")
    assert not is_abstain("根据 [1]，采购金额需二级授权。")


def test_summary_skips_abstain_when_not_measured():
    rows = [{"hit": 1.0, "rr": 1.0, "answerable": True, "cite_valid": 0, "cite_total": 0}]
    summary = summarize(rows, 5)
    assert summary["hit"] == 1.0
    assert summary["abstain_accuracy"] is None


def test_summary_computes_abstain_accuracy():
    rows = [
        {"hit": 1.0, "rr": 1.0, "answerable": False, "abstained": True},
        {"hit": 1.0, "rr": 1.0, "answerable": False, "abstained": False},
    ]
    assert summarize(rows)["abstain_accuracy"] == 0.5


def test_golden_set_is_well_formed():
    questions = load_golden()
    assert len(questions) >= 15
    ids = [item["id"] for item in questions]
    assert len(ids) == len(set(ids))
    for item in questions:
        assert item["q"].strip()
        if item["answerable"]:
            assert item["pages"], f"{item['id']} 需要标注页码"
