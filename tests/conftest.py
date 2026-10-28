"""Shared fixtures. Tests must never touch the network or the real database."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag.config import settings  # noqa: E402


@pytest.fixture
def offline(monkeypatch):
    """Disable every paid channel so a test run is free, fast and deterministic."""
    monkeypatch.setattr(settings, "llm_api_key", "")
    monkeypatch.setattr(settings, "embed_api_key", "")
    monkeypatch.setattr(settings, "rerank_enabled", False)
    monkeypatch.setattr(settings, "ocr_enabled", False)
    return settings


@pytest.fixture
def sample_pdf() -> Path:
    path = ROOT / "data" / "samples" / "sample-3pages.pdf"
    assert path.exists(), "样例 PDF 丢失"
    return path


@pytest.fixture
def service(tmp_path):
    from rag.service import Service
    from rag.store.db import Database

    return Service(Database(tmp_path / "test.sqlite"))
