"""自动出题：解析容错、按页分散取样、失败不静默。"""

import json

import pytest

from eval import generate
from rag.index.library import Record


def record(idx, page, text="这一段讲的是付款账期，最长 30 天。", doc_id="d1", filename="政策.pdf"):
    return Record(doc_id=doc_id, filename=filename, idx=idx, text=text, page_from=page, page_to=page)


# ── 解析 ────────────────────────────────────────────────────


def test_parse_plain_array():
    items = generate._parse('[{"q": "付款账期最长是多久？", "answer": "30 天"}]')
    assert items == [{"q": "付款账期最长是多久？", "answer": "30 天"}]


def test_parse_fenced_json():
    text = '好的，题目如下：\n```json\n[{"q": "谁审批紧急采购？", "answer": "分管领导"}]\n```\n希望有帮助'
    assert generate._parse(text)[0]["q"] == "谁审批紧急采购？"


def test_parse_single_object():
    assert generate._parse('{"q": "哪些情况可以单一来源采购？"}')[0]["q"].startswith("哪些情况")


def test_parse_garbage_returns_empty():
    assert generate._parse("抱歉，我无法完成这个请求。") == []
    assert generate._parse("[{broken json") == []


def test_parse_drops_too_short_question():
    assert generate._parse('[{"q": "吗？"}]') == []


# ── 取样 ────────────────────────────────────────────────────


def test_sampling_spreads_across_pages():
    records = [record(index, page=1) for index in range(4)]   # 4 个块都在第 1 页
    records += [record(index + 4, page=2) for index in range(4)]
    picked = generate.sample_records(records, limit=2)
    assert len(picked) == 2
    assert [item.page_from for item in picked] == [1, 2], "先按页分散，不能全落在第 1 页"


def test_sampling_fills_up_when_pages_run_out():
    records = [record(index, page=1) for index in range(4)]
    picked = generate.sample_records(records, limit=3)
    assert len(picked) == 3
    assert {item.page_from for item in picked} == {1}


def test_sampling_skips_empty_chunks():
    records = [record(0, 1, text="   "), record(1, 2)]
    picked = generate.sample_records(records, limit=5)
    assert [item.idx for item in picked] == [1]


# ── 生成 ────────────────────────────────────────────────────


@pytest.fixture()
def fake_llm(monkeypatch):
    state = {"replies": [], "prompts": []}

    def chat(messages, **kwargs):
        state["prompts"].append(messages[0]["content"])
        reply = state["replies"].pop(0) if state["replies"] else "[]"
        return reply, {}

    monkeypatch.setattr(generate, "chat", chat)
    return state


def test_generate_questions_carries_page_from_chunk(fake_llm):
    fake_llm["replies"] = [
        json.dumps([{"q": "付款账期最长是多少天？", "answer": "30 天"}], ensure_ascii=False),
        json.dumps([{"q": "谁负责审批紧急采购？", "answer": "分管领导"}], ensure_ascii=False),
    ]
    records = [record(0, page=3), record(1, page=7)]
    bundle = generate.generate_questions(records, limit=2)

    assert bundle["failed"] == 0 and bundle["sampled"] == 2
    assert [item["pages"] for item in bundle["questions"]] == [[3], [7]]
    assert bundle["questions"][0]["doc"] == "政策.pdf"
    assert bundle["questions"][0]["id"] == "auto-01"
    assert bundle["questions"][0]["ground_truth"] == "30 天"
    assert bundle["questions"][0]["source_chunk"] == 0
    # 提示词里要带上页码和文件名，否则模型会编出别的文档的问题
    assert "第 3 页" in fake_llm["prompts"][0] and "政策.pdf" in fake_llm["prompts"][0]


def test_generate_counts_failures_instead_of_silently_dropping(fake_llm):
    fake_llm["replies"] = ["无法完成", json.dumps([{"q": "哪些情况可以例外？"}], ensure_ascii=False)]
    bundle = generate.generate_questions([record(0, page=1), record(1, page=2)], limit=2)
    assert bundle["failed"] == 1
    assert len(bundle["questions"]) == 1 and bundle["questions"][0]["pages"] == [2]


def test_generated_questions_feed_the_evaluator():
    """产物必须能被 eval.run_eval 直接吃下：字段名对得上，pages 是 int。"""
    from eval.run_eval import run_questions  # noqa: F401  (import 只为确认依赖方向)
    from eval.metrics import is_abstain, rank_of_first_hit

    item = {"pages": [3]}
    spans = [(3, 3), (5, 5)]
    assert rank_of_first_hit(spans, item["pages"], k=5) == 1
    assert is_abstain("资料中没有相关条款") is True
