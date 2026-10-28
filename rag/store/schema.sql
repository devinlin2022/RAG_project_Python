-- One row per document. `path` points at the uploaded copy under data/uploads.
CREATE TABLE IF NOT EXISTS docs (
    id          TEXT PRIMARY KEY,
    filename    TEXT NOT NULL,
    path        TEXT NOT NULL,
    pages       INTEGER NOT NULL DEFAULT 0,
    chars       INTEGER NOT NULL DEFAULT 0,
    created_at  TEXT NOT NULL,
    strategy    TEXT,
    chunk_count INTEGER NOT NULL DEFAULT 0,
    dim         INTEGER NOT NULL DEFAULT 0,
    indexed_at  TEXT,
    enhanced_at TEXT
);

-- Page text and the V2 quality flags, so a restart keeps the page table intact.
CREATE TABLE IF NOT EXISTS pages (
    doc_id       TEXT NOT NULL,
    page         INTEGER NOT NULL,
    text         TEXT NOT NULL,
    chars        INTEGER NOT NULL,
    density      REAL NOT NULL DEFAULT 0,
    tables       INTEGER NOT NULL DEFAULT 0,
    image_cover  REAL NOT NULL DEFAULT 0,
    vector_ops   INTEGER NOT NULL DEFAULT 0,
    column_ratio REAL NOT NULL DEFAULT 0,
    scan_like    INTEGER NOT NULL DEFAULT 0,
    table_like   INTEGER NOT NULL DEFAULT 0,
    figure_like  INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (doc_id, page)
);

-- Enhanced blocks (OCR / table / figure). Persisted so vision calls are paid once.
CREATE TABLE IF NOT EXISTS blocks (
    doc_id TEXT NOT NULL,
    page   INTEGER NOT NULL,
    ord    INTEGER NOT NULL,
    type   TEXT NOT NULL,
    text   TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (doc_id, page, ord)
);

-- Chunks carry their own vector: deleting a document is one DELETE, not a
-- rebuild of a separate index file that can drift out of sync.
CREATE TABLE IF NOT EXISTS chunks (
    doc_id    TEXT NOT NULL,
    idx       INTEGER NOT NULL,
    text      TEXT NOT NULL,
    page_from INTEGER NOT NULL,
    page_to   INTEGER NOT NULL,
    kind      TEXT NOT NULL DEFAULT 'text',
    section   TEXT NOT NULL DEFAULT '',
    level     INTEGER NOT NULL DEFAULT 0,
    chars     INTEGER NOT NULL DEFAULT 0,
    vec       BLOB,
    PRIMARY KEY (doc_id, idx)
);

CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks (doc_id);

CREATE TABLE IF NOT EXISTS queries (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    question   TEXT NOT NULL,
    rewritten  TEXT NOT NULL DEFAULT '',
    answer     TEXT NOT NULL DEFAULT '',
    pages      TEXT NOT NULL DEFAULT '',
    ms         INTEGER NOT NULL DEFAULT 0,
    top_k      INTEGER NOT NULL DEFAULT 0,
    strategy   TEXT NOT NULL DEFAULT '',
    reranked   INTEGER NOT NULL DEFAULT 0
);
