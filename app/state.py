"""In-memory document store shared by UI and API. Replaced by SQLite in V5."""

from __future__ import annotations


class DocStore:
    def __init__(self) -> None:
        self._docs: dict[str, dict] = {}

    def put(self, doc: dict) -> dict:
        self._docs[doc["id"]] = doc
        return doc

    def get(self, doc_id: str | None) -> dict | None:
        if not doc_id:
            return None
        return self._docs.get(doc_id)

    def latest(self) -> dict | None:
        if not self._docs:
            return None
        return list(self._docs.values())[-1]

    def all(self) -> list[dict]:
        return list(self._docs.values())

    def __len__(self) -> int:
        return len(self._docs)


STORE = DocStore()
