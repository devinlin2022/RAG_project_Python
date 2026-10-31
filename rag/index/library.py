"""In-memory chunk library: text records, vector index and BM25 index.

Loaded from SQLite at startup, mutated on ingest, delete and re-index. Reads
are lock-free; writes take a lock so a rebuild cannot interleave with a query.
"""

import logging
import threading
from dataclasses import dataclass, field

import numpy as np

from rag.config import settings
from rag.index.bm25 import BM25
from rag.index.dense import FlatIndex
from rag.index.tokenize import tokenize
from rag.store.db import Database

log = logging.getLogger(__name__)


@dataclass
class Record:
    doc_id: str
    filename: str
    idx: int
    text: str
    page_from: int
    page_to: int
    kind: str = "text"
    section: str = ""
    level: int = 0
    tokens: list[str] = field(default_factory=list)
    vec: np.ndarray | None = None

    @property
    def chars(self) -> int:
        """Same surface as `Chunk`, so one renderer can draw either."""
        return len(self.text)

    @property
    def page_label(self) -> str:
        if self.page_from == self.page_to:
            return f"p.{self.page_from}"
        return f"p.{self.page_from}-{self.page_to}"


class ChunkLibrary:
    def __init__(self, db: Database) -> None:
        self.db = db
        self.records: list[Record] = []
        self.dense = FlatIndex()
        self.bm25 = BM25([])
        self._dense_rows: list[int] = []
        self._dense_doc_ids: np.ndarray = np.array([], dtype=object)
        self._write_lock = threading.Lock()
        self.load()

    # ---- lifecycle ----

    def load(self) -> None:
        rows = self.db.load_chunks()
        self.records = [
            Record(
                doc_id=row["doc_id"],
                filename=row["filename"],
                idx=row["idx"],
                text=row["text"],
                page_from=row["page_from"],
                page_to=row["page_to"],
                kind=row["kind"],
                section=row["section"],
                level=row["level"],
                tokens=tokenize(row["text"]),
                vec=row["vec"],
            )
            for row in rows
        ]
        self._rebuild()
        log.info("library loaded: %d chunks, %d vectors", len(self.records), len(self.dense))

    def _rebuild(self) -> None:
        self.bm25 = BM25([record.tokens for record in self.records])
        vectors = [(i, r.vec) for i, r in enumerate(self.records) if r.vec is not None]
        if vectors:
            matrix = np.vstack([vec for _, vec in vectors])
            self.dense = FlatIndex(matrix)
            self._dense_rows = [i for i, _ in vectors]
            self._dense_doc_ids = np.array(
                [self.records[i].doc_id for i, _ in vectors], dtype=object
            )
        else:
            self.dense = FlatIndex()
            self._dense_rows = []
            self._dense_doc_ids = np.array([], dtype=object)

    def replace_document(self, doc_id: str, chunks: list[dict]) -> int:
        """Persist and load one document's chunks. `chunks` may carry `vec`."""
        with self._write_lock:
            self.db.replace_chunks(doc_id, chunks)
            doc = self.db.get_doc(doc_id) or {}
            filename = doc.get("filename") or doc_id
            self.records = [r for r in self.records if r.doc_id != doc_id]
            for chunk in chunks:
                self.records.append(
                    Record(
                        doc_id=doc_id,
                        filename=filename,
                        idx=chunk["idx"],
                        text=chunk["text"],
                        page_from=chunk["page_from"],
                        page_to=chunk["page_to"],
                        kind=chunk.get("kind", "text"),
                        section=chunk.get("section", ""),
                        level=int(chunk.get("level", 0)),
                        tokens=tokenize(chunk["text"]),
                        vec=chunk.get("vec"),
                    )
                )
            self.records.sort(key=lambda r: (r.doc_id, r.idx))
            self._rebuild()
            return len(chunks)

    def remove_document(self, doc_id: str) -> None:
        with self._write_lock:
            self.db.delete_doc(doc_id)
            self.records = [r for r in self.records if r.doc_id != doc_id]
            self._rebuild()

    # ---- search ----

    @property
    def embed_model(self) -> str:
        return settings.embed_model

    def _mask(self, doc_ids: list[str] | None) -> np.ndarray | None:
        if not doc_ids:
            return None
        return np.isin(self._dense_doc_ids, list(doc_ids))

    def dense_search(
        self, vector: list[float] | np.ndarray, k: int, doc_ids: list[str] | None = None
    ) -> list[tuple[int, float]]:
        """Return (record index, cosine) pairs."""
        if len(self.dense) == 0:
            return []
        allowed = self._mask(doc_ids)
        hits = self.dense.search(vector, k, allowed)
        return [(self._dense_rows[row], score) for row, score in hits]

    def sparse_search(
        self, query: str, k: int, doc_ids: list[str] | None = None
    ) -> list[tuple[int, float]]:
        tokens = tokenize(query)
        if not tokens or not self.records:
            return []
        scores = self.bm25.scores(tokens)
        order = np.argsort(-np.asarray(scores))[: max(k * 4, k)]
        allowed = set(doc_ids) if doc_ids else None
        hits: list[tuple[int, float]] = []
        for row in order:
            score = float(scores[row])
            if score <= 0:
                break
            record = self.records[int(row)]
            if allowed and record.doc_id not in allowed:
                continue
            hits.append((int(row), score))
            if len(hits) >= k:
                break
        return hits

    def stats(self) -> dict:
        docs = {r.doc_id for r in self.records}
        return {
            "chunks": len(self.records),
            "docs": len(docs),
            "vectors": len(self.dense),
            "dim": self.dense.dim,
            "vocab": len(self.bm25.idf),
        }
