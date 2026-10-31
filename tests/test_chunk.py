"""Chunking tests. The contract fixture is the acceptance case: clauses must
survive whole and chunks must not be tiny."""

import pytest

from rag.chunk.metrics import clause_integrity, composite
from rag.chunk.profile import profile_document
from rag.chunk.registry import chunk_document, compare_strategies
from rag.chunk.semantic import semantic_split, split_sentences
from rag.chunk.structure import build_sections, parse_heading
from rag.ingest.enhance import Block

CONTRACT = """第一章 总则

第一条 合同目的
甲方委托乙方提供智能文档问答系统的开发与部署服务，乙方接受委托并按本合同约定完成交付。

第二条 服务范围
2.1 乙方应完成以下工作：
（一）完成文档解析与切分模块的开发和测试；
（二）完成检索与重排模块的集成；
（三）提供不少于两次的现场培训。
2.2 乙方不得将本合同项下义务整体转委托给第三方，经甲方书面同意的分包除外。

第二章 付款与交付

第三条 付款方式
3.1 合同总价为人民币壹拾万元整，含税。
3.2 甲方逾期付款的，应按日万分之五向乙方支付违约金。

第四条 交付与验收
4.1 乙方应于合同生效之日起六十日内完成交付。
4.2 甲方应在收到交付物后十五个工作日内完成验收，逾期未提出书面异议的，视为验收通过。

第三章 违约责任

第五条 保密
5.1 双方对因履行本合同知悉的对方商业秘密负有保密义务，保密期限为合同终止后三年。
5.2 违反保密义务的一方应赔偿对方因此遭受的全部损失。

第六条 争议解决
6.1 因本合同引起的争议，双方应友好协商解决。
6.2 协商不成的，提交合同签订地有管辖权的人民法院诉讼解决。
"""


def pairs_of(text: str, page: int = 1):
    return [(page, Block("text", text, "pdfminer"))]


@pytest.fixture
def contract_pairs():
    return pairs_of(CONTRACT)


class TestHeadingParser:
    def test_article_and_chapter_levels(self):
        assert parse_heading("第一章 总则")[0] == 1
        assert parse_heading("第一条 合同目的")[0] == 3

    def test_decimal_nests_by_dot_count(self):
        assert parse_heading("2.1 服务内容")[0] == 5
        assert parse_heading("2.1.3 细则")[0] == 6

    def test_decimal_outranks_circled_chinese(self):
        assert parse_heading("（一）完成开发")[0] > parse_heading("2.1 服务内容")[0]

    def test_body_sentence_is_not_a_heading(self):
        assert parse_heading("甲方委托乙方提供系统开发服务。") is None

    def test_table_row_is_not_a_heading(self):
        assert parse_heading("| 节点 | 比例 |") is None

    def test_markdown_heading(self):
        level, label = parse_heading("## 付款方式")
        assert level == 2
        assert label == "付款方式"


class TestSections:
    def test_path_carries_ancestry(self, contract_pairs):
        sections = build_sections(contract_pairs)
        paths = [s.path for s in sections]
        assert any("第二章 付款与交付 > 第三条 付款方式 > 3.1" in p for p in paths)

    def test_section_spans_are_monotonic(self, contract_pairs):
        sections = build_sections(contract_pairs)
        assert all(s.page_from <= s.page_to for s in sections)


class TestStructuredChunking:
    def test_clauses_survive_whole(self, contract_pairs):
        result = chunk_document(contract_pairs, "hybrid", size=500, overlap=0, pages=1)
        assert clause_integrity(CONTRACT, result["chunks"]) == 1.0

    def test_no_micro_chunks(self, contract_pairs):
        result = chunk_document(contract_pairs, "hybrid", size=500, overlap=0, pages=1)
        average = sum(c.chars for c in result["chunks"]) / len(result["chunks"])
        assert average >= 60, f"average chunk too small: {average:.1f}"

    def test_chunk_never_spans_two_chapters(self, contract_pairs):
        result = chunk_document(contract_pairs, "hybrid", size=500, overlap=0, pages=1)
        for chunk in result["chunks"]:
            assert "第一章" not in chunk.text or "第二章" not in chunk.text

    def test_tables_keep_kind_and_header(self):
        table = Block(
            "table",
            "[表 1 · 第 1 页]\n| 节点 | 比例 |\n| --- | --- |\n"
            + "\n".join(f"| 节点{i} | {i}0% |" for i in range(30)),
            "pdfplumber",
        )
        pairs = [(1, Block("text", "第三条 付款方式\n3.2 付款节点如下：", "pdfminer")), (1, table)]
        result = chunk_document(pairs, "structured", size=200, overlap=0, pages=1)
        tables = [c for c in result["chunks"] if c.kind == "table"]
        assert len(tables) > 1, "wide table should be split by rows"
        assert all("| 节点 | 比例 |" in c.text for c in tables), "header must repeat"


class TestSemanticChunking:
    def test_sentence_split_keeps_decimals(self):
        sentences = split_sentences("合同总价 3.5 万元。第二句在这里！")
        assert sentences[0] == "合同总价 3.5 万元。"

    def test_percentile_breakpoints_used(self, monkeypatch):
        import rag.chunk.semantic as semantic

        def fake_embed(texts, batch=None):
            # Two topical clusters: first half similar, second half similar.
            return [[1.0, 0.0] if i < len(texts) / 2 else [0.0, 1.0] for i in range(len(texts))]

        monkeypatch.setattr(semantic, "embed_texts", fake_embed)
        text = "".join(f"第{i}句关于付款的约定。" for i in range(4))
        text += "".join(f"第{i}句关于保密的约定。" for i in range(4))
        pieces, info = semantic_split(text, size=70, min_chars=10, percentile=80)
        assert info["mode"] == "percentile"
        assert info["breakpoints"] >= 1, "a clear topic change must create a breakpoint"
        assert len(pieces) >= 2

    def test_degrades_without_key(self, contract_pairs, monkeypatch):
        import rag.chunk.semantic as semantic
        from rag.embed.client import EmbedError

        def boom(texts, batch=None):
            raise EmbedError("no key")

        monkeypatch.setattr(semantic, "embed_texts", boom)
        pieces, info = semantic_split(CONTRACT, size=300, min_chars=80, percentile=85)
        assert info["mode"] == "fallback"
        assert pieces


class TestProfiler:
    def test_contract_recommends_hybrid(self, contract_pairs):
        profile = profile_document(contract_pairs, pages=1)
        assert profile.strategy == "hybrid"
        assert profile.overlap == 0
        assert profile.reasons

    def test_short_prose_recommends_recursive(self):
        profile = profile_document(pairs_of("这是一段很短的说明文字，没有任何编号结构。"), pages=1)
        assert profile.strategy == "recursive"

    def test_profile_is_fast(self, contract_pairs):
        import time

        started = time.perf_counter()
        profile_document(contract_pairs, pages=1)
        assert (time.perf_counter() - started) < 0.5


class TestComparison:
    def test_table_has_all_four_strategies(self, contract_pairs):
        rows = compare_strategies(contract_pairs, size=500, overlap=0, pages=1)
        assert {row["strategy"] for row in rows} == {"hybrid", "structured", "semantic", "recursive"}
        assert rows[0]["recommended"] is True

    def test_hybrid_beats_recursive_on_structure(self, contract_pairs):
        rows = {row["strategy"]: row for row in compare_strategies(contract_pairs, size=500, pages=1)}
        assert rows["hybrid"]["struct_score"] >= rows["recursive"]["struct_score"]
        assert rows["hybrid"]["section_coverage"] > rows["recursive"]["section_coverage"]


class TestTocStubs:

    TOC = """目录

第一章 总则
第二章 供应商准入与评估
第三章 采购申请与审批
第四章 发票校验与付款

第一章 总则

第一条 目的
本办法适用于示例组织的全部采购活动。
"""

    def test_toc_entries_do_not_become_chunks(self):
        pairs = [(1, Block(type="text", text=self.TOC, source="text"))]
        result = chunk_document(pairs, "hybrid", size=500, pages=1)
        assert all(len(chunk.text) > 30 for chunk in result["chunks"])
        assert any("适用于示例组织" in chunk.text for chunk in result["chunks"])

    def test_a_single_short_chunk_survives(self):
        text = "第一章 总则\n\n第一条 目的\n本办法适用于示例组织的全部采购活动。"
        pairs = [(1, Block(type="text", text=text, source="text"))]
        result = chunk_document(pairs, "hybrid", size=500, pages=1)
        assert result["chunks"]


class TestMetrics:
    def test_composite_keys(self, contract_pairs):
        result = chunk_document(contract_pairs, "hybrid", size=500, pages=1)
        metrics = composite(result["chunks"], CONTRACT, 120, 500)
        for key in (
            "chunks",
            "avg_chars",
            "chunks_per_1k",
            "boundary_violation",
            "clause_integrity",
            "size_conformance",
            "section_coverage",
            "overlap_waste",
            "struct_score",
        ):
            assert key in metrics
