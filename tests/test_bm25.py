"""Tokenizer and BM25: the lexical rail must match codes, not syllables."""

from rag.index.bm25 import BM25
from rag.index.tokenize import tokenize


def test_latin_codes_stay_whole():
    tokens = tokenize("编号 INV-2048 与 PO-77 建立映射，规则 FLOW9")
    assert "inv-2048" in tokens
    assert "po-77" in tokens
    assert "flow9" in tokens
    assert "inv" not in tokens


def test_chinese_terms_are_segmented():
    tokens = tokenize("采购订单的释放策略")
    assert any("采购" in token for token in tokens)
    assert "的" not in tokens


def test_numbers_survive():
    assert "29" in tokenize("共 29 个定价过程")


def test_bm25_ranks_the_matching_document_first():
    corpus = [
        tokenize("付款账期最长为六十日"),
        tokenize("库存上传使用 VALIDATE_STOCK 校验"),
        tokenize("预付款比例不超过百分之二十"),
    ]
    index = BM25(corpus)
    scores = index.scores(tokenize("VALIDATE_STOCK 校验"))
    assert scores.index(max(scores)) == 1


def test_bm25_unknown_terms_score_zero():
    index = BM25([tokenize("采购订单"), tokenize("供应商")])
    assert max(index.scores(tokenize("完全不存在 的 词"))) == 0.0


def test_empty_corpus_is_safe():
    index = BM25([])
    assert index.scores(["x"]) == []
