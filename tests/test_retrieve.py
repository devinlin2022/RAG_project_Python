"""Retrieval wiring: rail agreement, rerank fallback, page attribution."""

import numpy as np

from rag.index.library import ChunkLibrary
from rag.index.fusion import rrf
from rag.retrieve.retriever import rerank_hits, search
from rag.store.db import Database


def _library(tmp_path) -> ChunkLibrary:
    db = Database(tmp_path / "lib.sqlite")
    db.upsert_doc({"id": "d1", "filename": "a.pdf", "path": "p", "pages": 3, "chars": 30})
    chunks = [
        {"idx": 0, "text": "付款账期最长为六十日", "page_from": 1, "page_to": 1,
         "vec": np.array([1.0, 0.0, 0.0], dtype="float32")},
        {"idx": 1, "text": "库存上传用 VALIDATE_STOCK 校验", "page_from": 2, "page_to": 2,
         "vec": np.array([0.0, 1.0, 0.0], dtype="float32")},
        {"idx": 2, "text": "预付款比例不超过百分之二十", "page_from": 3, "page_to": 3,
         "vec": np.array([0.0, 0.0, 1.0], dtype="float32")},
    ]
    lib = ChunkLibrary(db)
    lib.replace_document("d1", chunks)
    return lib


def test_dense_search_orders_by_similarity(tmp_path):
    lib = _library(tmp_path)
    hits = lib.dense_search([0.9, 0.1, 0.0], 2)
    assert hits[0][0] == 0


def test_sparse_search_matches_a_code(tmp_path):
    lib = _library(tmp_path)
    hits = lib.sparse_search("VALIDATE_STOCK", 2)
    assert lib.records[hits[0][0]].page_from == 2


def test_doc_filter_excludes_other_documents(tmp_path):
    lib = _library(tmp_path)
    assert lib.sparse_search("预付款", 5, doc_ids=["nope"]) == []


def test_rrf_search_includes_both_rails(tmp_path):
    lib = _library(tmp_path)
    hits, info = search("预付款 百分之二十", [1.0, 0.0, 0.0], lib, strategy="rrf", top_k=3)
    assert info["dense_hits"] == 3
    assert info["sparse_hits"] >= 1
    pages = {hit.record.page_from for hit in hits}
    assert 1 in pages and 3 in pages


def test_rerank_reorders_hits(tmp_path):
    lib = _library(tmp_path)
    hits, _ = search("任意", [0.8, 0.5, 0.1], lib, strategy="dense", top_k=3, candidates=3)
    calls = []

    def fake_reranker(query, documents):
        calls.append(len(documents))
        # reverse the fusion order on purpose
        return [(index, float(index)) for index in range(len(documents))]

    reranked, info = rerank_hits("q", hits, top_k=2, reranker=fake_reranker)
    assert info["used"] is True
    assert calls == [3]
    assert [hit.record.page_from for hit in reranked] == [3, 2]


def test_broken_reranker_keeps_fusion_order(tmp_path):
    lib = _library(tmp_path)
    hits, _ = search("任意", [1.0, 0.0, 0.0], lib, strategy="dense", top_k=3)

    def broken(query, documents):
        raise RuntimeError("402 余额不足")

    kept, info = rerank_hits("q", hits, top_k=2, reranker=broken)
    assert info["used"] is False
    assert "402" in info["error"]
    assert len(kept) == 2


def test_rrf_scores_are_rank_based():
    fused = rrf([[0, 1], [1, 0]], k=60)
    assert abs(fused[0] - fused[1]) < 1e-12
