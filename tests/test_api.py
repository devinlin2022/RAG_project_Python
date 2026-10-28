"""HTTP contract: status codes, auth gate, and the abstain path with an empty index."""

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.api import public, router  # noqa: E402
from app.state import set_service  # noqa: E402
from rag.config import VERSION  # noqa: E402
from rag.service import Service  # noqa: E402
from rag.store.db import Database  # noqa: E402


@pytest.fixture
def client(tmp_path, offline):
    service = Service(Database(tmp_path / "api.sqlite"))
    set_service(service)
    app = FastAPI()
    app.include_router(public)
    app.include_router(router)
    yield TestClient(app), service
    set_service(None)


def test_health_is_public(client):
    http, _ = client
    body = http.get("/health").json()
    assert body["status"] == "ok" and body["version"] == VERSION


def test_docs_is_empty_before_ingest(client):
    http, _ = client
    assert http.get("/documents").json() == []


def test_ask_without_index_abstains(client):
    http, _ = client
    response = http.post("/ask", json={"question": "付款账期是多久？"})
    assert response.status_code == 200
    body = response.json()
    assert body["abstained"] is True
    assert body["sources"] == []


def test_unknown_doc_returns_404(client):
    http, _ = client
    assert http.get("/documents/nope/pages").status_code == 404
    assert http.delete("/documents/nope").status_code == 404


def test_missing_question_returns_400(client):
    http, _ = client
    assert http.post("/ask", json={"question": "   "}).status_code == 400


def test_api_key_gate(client, monkeypatch):
    http, _ = client
    monkeypatch.setattr("app.auth.settings.app_api_key", "secret")
    assert http.get("/health").status_code == 200
    assert http.post("/ask", json={"question": "x"}).status_code == 401
    assert http.post(
        "/ask", json={"question": "x"}, headers={"X-API-Key": "secret"}
    ).status_code == 200


def test_retrieve_returns_all_three_rails(client):
    http, _ = client
    body = http.post("/retrieve", json={"query": "采购"}).json()
    assert set(body["rails"]) == {"dense", "sparse", "rrf"}
    assert body["rails"]["rrf"] == []


def test_ingest_rejects_non_pdf(client):
    http, _ = client
    response = http.post("/ingest", files={"file": ("a.txt", b"hello", "text/plain")})
    assert response.status_code == 400
