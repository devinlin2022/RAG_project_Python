"""SQLite persistence in WAL mode: documents, pages, blocks, chunks, queries."""

import json
import logging
import sqlite3
import threading
from datetime import datetime
from pathlib import Path

import numpy as np

from rag.config import BASE_DIR

log = logging.getLogger(__name__)

_SCHEMA = Path(__file__).with_name("schema.sql")
_write_lock = threading.Lock()


class Database:
    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else BASE_DIR / "data" / "rag.sqlite"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.init()

    # Connection per call keeps the module thread-safe: Gradio runs handlers in
    # a thread pool and SQLite connections are not shareable across threads.
    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=15)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def init(self) -> None:
        with _write_lock, self._connect() as conn:
            conn.executescript(_SCHEMA.read_text(encoding="utf-8"))
            # CREATE TABLE IF NOT EXISTS leaves an existing database alone, so
            # columns added later are patched in here instead of asking the user
            # to delete their index.
            self._add_missing_columns(
                conn,
                {
                    "docs": {"format": "TEXT NOT NULL DEFAULT ''"},
                    "queries": {
                        "tokens_in": "INTEGER NOT NULL DEFAULT 0",
                        "tokens_out": "INTEGER NOT NULL DEFAULT 0",
                    },
                },
            )

    @staticmethod
    def _add_missing_columns(conn, wanted: dict[str, dict[str, str]]) -> None:
        for table, columns in wanted.items():
            existing = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
            if not existing:
                continue
            for name, ddl in columns.items():
                if name not in existing:
                    conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {ddl}")

    # ---- documents ----

    def upsert_doc(self, meta: dict) -> None:
        with _write_lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO docs (id, filename, path, format, pages, chars, created_at,
                                  strategy, chunk_count, dim, indexed_at, enhanced_at)
                VALUES (:id, :filename, :path, :format, :pages, :chars, :created_at,
                        :strategy, :chunk_count, :dim, :indexed_at, :enhanced_at)
                ON CONFLICT(id) DO UPDATE SET
                    filename=excluded.filename, path=excluded.path, format=excluded.format,
                    pages=excluded.pages, chars=excluded.chars,
                    strategy=excluded.strategy, chunk_count=excluded.chunk_count,
                    dim=excluded.dim, indexed_at=excluded.indexed_at,
                    enhanced_at=excluded.enhanced_at
                """,
                {
                    "id": meta["id"],
                    "filename": meta["filename"],
                    "path": meta["path"],
                    "format": meta.get("format", ""),
                    "pages": int(meta.get("pages", 0)),
                    "chars": int(meta.get("chars", 0)),
                    "created_at": meta.get("created_at") or datetime.now().isoformat(timespec="seconds"),
                    "strategy": meta.get("strategy"),
                    "chunk_count": int(meta.get("chunk_count", 0)),
                    "dim": int(meta.get("dim", 0)),
                    "indexed_at": meta.get("indexed_at"),
                    "enhanced_at": meta.get("enhanced_at"),
                },
            )

    def get_doc(self, doc_id: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM docs WHERE id = ?", (doc_id,)).fetchone()
        return dict(row) if row else None

    def list_docs(self) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM docs ORDER BY created_at").fetchall()
        return [dict(row) for row in rows]

    def delete_doc(self, doc_id: str) -> bool:
        with _write_lock, self._connect() as conn:
            for table in ("chunks", "pages", "blocks"):
                conn.execute(f"DELETE FROM {table} WHERE doc_id = ?", (doc_id,))
            cursor = conn.execute("DELETE FROM docs WHERE id = ?", (doc_id,))
        return cursor.rowcount > 0

    def mark_enhanced(self, doc_id: str) -> None:
        with _write_lock, self._connect() as conn:
            conn.execute(
                "UPDATE docs SET enhanced_at = ? WHERE id = ?",
                (datetime.now().isoformat(timespec="seconds"), doc_id),
            )

    # ---- pages ----

    def save_pages(self, doc_id: str, pages: list[dict]) -> None:
        rows = [
            (
                doc_id,
                p["page"],
                p["text"],
                p["chars"],
                p.get("density", 0.0),
                p.get("tables", 0),
                p.get("image_cover", 0.0),
                p.get("vector_ops", 0),
                p.get("column_ratio", 0.0),
                int(bool(p.get("scan_like"))),
                int(bool(p.get("table_like"))),
                int(bool(p.get("figure_like"))),
            )
            for p in pages
        ]
        with _write_lock, self._connect() as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO pages (doc_id, page, text, chars, density, tables,"
                " image_cover, vector_ops, column_ratio, scan_like, table_like, figure_like)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                rows,
            )

    def load_pages(self, doc_id: str) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM pages WHERE doc_id = ? ORDER BY page", (doc_id,)
            ).fetchall()
        return [
            {
                "page": row["page"],
                "text": row["text"],
                "chars": row["chars"],
                "density": row["density"],
                "tables": row["tables"],
                "image_cover": row["image_cover"],
                "vector_ops": row["vector_ops"],
                "column_ratio": row["column_ratio"],
                "scan_like": bool(row["scan_like"]),
                "table_like": bool(row["table_like"]),
                "figure_like": bool(row["figure_like"]),
            }
            for row in rows
        ]

    # ---- enhanced blocks ----

    def save_blocks(self, doc_id: str, blocks_by_page: dict[int, list]) -> None:
        rows = [
            (doc_id, page, order, block.type, block.text, block.source)
            for page, blocks in blocks_by_page.items()
            for order, block in enumerate(blocks)
        ]
        with _write_lock, self._connect() as conn:
            conn.execute("DELETE FROM blocks WHERE doc_id = ?", (doc_id,))
            conn.executemany(
                "INSERT INTO blocks (doc_id, page, ord, type, text, source) VALUES (?,?,?,?,?,?)",
                rows,
            )

    def load_blocks(self, doc_id: str) -> dict[int, list[dict]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT page, type, text, source FROM blocks WHERE doc_id = ? ORDER BY page, ord",
                (doc_id,),
            ).fetchall()
        out: dict[int, list[dict]] = {}
        for row in rows:
            out.setdefault(row["page"], []).append(
                {"type": row["type"], "text": row["text"], "source": row["source"]}
            )
        return out

    # ---- chunks ----

    def replace_chunks(self, doc_id: str, chunks: list[dict]) -> None:
        rows = [
            (
                doc_id,
                chunk["idx"],
                chunk["text"],
                chunk["page_from"],
                chunk["page_to"],
                chunk.get("kind", "text"),
                chunk.get("section", ""),
                int(chunk.get("level", 0)),
                len(chunk["text"]),
                _to_blob(chunk.get("vec")),
            )
            for chunk in chunks
        ]
        with _write_lock, self._connect() as conn:
            conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
            conn.executemany(
                "INSERT INTO chunks (doc_id, idx, text, page_from, page_to, kind, section,"
                " level, chars, vec) VALUES (?,?,?,?,?,?,?,?,?,?)",
                rows,
            )

    def load_chunks(self, doc_id: str | None = None) -> list[dict]:
        sql = (
            "SELECT c.*, d.filename FROM chunks c JOIN docs d ON d.id = c.doc_id"
            + (" WHERE c.doc_id = ?" if doc_id else "")
            + " ORDER BY c.doc_id, c.idx"
        )
        with self._connect() as conn:
            rows = conn.execute(sql, (doc_id,) if doc_id else ()).fetchall()
        return [
            {
                "doc_id": row["doc_id"],
                "filename": row["filename"],
                "idx": row["idx"],
                "text": row["text"],
                "page_from": row["page_from"],
                "page_to": row["page_to"],
                "kind": row["kind"],
                "section": row["section"],
                "level": row["level"],
                "chars": row["chars"],
                "vec": _from_blob(row["vec"]),
            }
            for row in rows
        ]

    # ---- queries ----

    def log_query(
        self,
        question: str,
        answer: str,
        pages: list[int],
        ms: int,
        top_k: int,
        strategy: str,
        rewritten: str = "",
        reranked: bool = False,
        tokens_in: int = 0,
        tokens_out: int = 0,
    ) -> None:
        with _write_lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO queries (created_at, question, rewritten, answer, pages, ms,"
                " top_k, strategy, reranked, tokens_in, tokens_out)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    datetime.now().isoformat(timespec="seconds"),
                    question[:500],
                    rewritten[:500],
                    answer[:4000],
                    json.dumps(pages),
                    ms,
                    top_k,
                    strategy,
                    int(reranked),
                    int(tokens_in or 0),
                    int(tokens_out or 0),
                ),
            )

    def usage_totals(self) -> dict:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS requests,"
                " COALESCE(SUM(tokens_in), 0) AS llm_in,"
                " COALESCE(SUM(tokens_out), 0) AS llm_out FROM queries"
            ).fetchone()
        return {
            "requests": int(row["requests"] or 0),
            "llm_in": int(row["llm_in"] or 0),
            "llm_out": int(row["llm_out"] or 0),
            "embed": 0,
        }

    def recent_queries(self, limit: int = 20) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM queries ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(row) for row in rows]

    def stats(self) -> dict:
        with self._connect() as conn:
            docs = conn.execute("SELECT COUNT(*) FROM docs").fetchone()[0]
            chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            vectors = conn.execute("SELECT COUNT(*) FROM chunks WHERE vec IS NOT NULL").fetchone()[0]
            queries = conn.execute("SELECT COUNT(*) FROM queries").fetchone()[0]
            size = self.path.stat().st_size if self.path.exists() else 0
        return {
            "docs": docs,
            "chunks": chunks,
            "vectors": vectors,
            "queries": queries,
            "db_bytes": size,
        }


def _to_blob(vector) -> bytes | None:
    if vector is None:
        return None
    array = np.asarray(vector, dtype="float32")
    return array.tobytes() if array.size else None


def _from_blob(blob) -> np.ndarray | None:
    if not blob:
        return None
    return np.frombuffer(blob, dtype="float32")
