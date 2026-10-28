"""SQLite persistence: pages, blocks, chunks with vectors, deletion cascade."""

import numpy as np

from rag.store.db import Database


def _db(tmp_path) -> Database:
    return Database(tmp_path / "unit.sqlite")


def test_documents_roundtrip(tmp_path):
    db = _db(tmp_path)
    db.upsert_doc(
        {"id": "d1", "filename": "a.pdf", "path": "/tmp/a.pdf", "pages": 2, "chars": 10}
    )
    doc = db.get_doc("d1")
    assert doc["filename"] == "a.pdf"
    assert [row["id"] for row in db.list_docs()] == ["d1"]


def test_upsert_updates_index_fields(tmp_path):
    db = _db(tmp_path)
    db.upsert_doc({"id": "d1", "filename": "a.pdf", "path": "p", "pages": 1, "chars": 5})
    db.upsert_doc(
        {
            "id": "d1", "filename": "a.pdf", "path": "p", "pages": 1, "chars": 5,
            "strategy": "structured", "chunk_count": 4, "dim": 1024, "indexed_at": "now",
        }
    )
    doc = db.get_doc("d1")
    assert doc["chunk_count"] == 4 and doc["dim"] == 1024


def test_pages_and_blocks_survive_reload(tmp_path):
    db = _db(tmp_path)
    db.upsert_doc({"id": "d1", "filename": "a.pdf", "path": "p", "pages": 1, "chars": 3})
    db.save_pages("d1", [{"page": 1, "text": "abc", "chars": 3, "scan_like": True}])
    pages = db.load_pages("d1")
    assert pages[0]["text"] == "abc" and pages[0]["scan_like"] is True

    from rag.ingest.enhance import Block

    db.save_blocks("d1", {1: [Block("table", "| a | b |", "pdfplumber")]})
    blocks = db.load_blocks("d1")
    assert blocks[1][0]["type"] == "table"


def test_vectors_are_stored_and_reloaded(tmp_path):
    db = _db(tmp_path)
    db.upsert_doc({"id": "d1", "filename": "a.pdf", "path": "p", "pages": 1, "chars": 3})
    vector = [0.1, 0.2, 0.3]
    db.replace_chunks(
        "d1",
        [{"idx": 0, "text": "abc", "page_from": 1, "page_to": 1, "vec": vector}],
    )
    rows = db.load_chunks()
    assert np.allclose(rows[0]["vec"], np.array(vector, dtype="float32"))


def test_delete_cascades(tmp_path):
    db = _db(tmp_path)
    db.upsert_doc({"id": "d1", "filename": "a.pdf", "path": "p", "pages": 1, "chars": 3})
    db.save_pages("d1", [{"page": 1, "text": "abc", "chars": 3}])
    db.replace_chunks("d1", [{"idx": 0, "text": "abc", "page_from": 1, "page_to": 1}])
    assert db.delete_doc("d1") is True
    assert db.load_chunks() == []
    assert db.load_pages("d1") == []


def test_query_log(tmp_path):
    db = _db(tmp_path)
    db.log_query("q", "a", [1, 2], 120, 3, "rrf")
    rows = db.recent_queries(5)
    assert rows[0]["question"] == "q" and rows[0]["pages"] == "[1, 2]"
