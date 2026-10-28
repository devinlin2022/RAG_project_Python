"""Hardening: secret masking, JSON logs, rate limiting, API key gate."""

import json
import logging

import pytest
from fastapi import HTTPException

from app.auth import JsonFormatter, RateLimiter, mask, require_key
from rag.config import settings


def test_mask_removes_api_keys():
    text = "failed with key sk-demo0123456789abcdefghijklmnopqrstuv"
    assert "sk-demo" not in mask(text)
    assert "sk-***" in mask(text)


def test_mask_removes_bearer_headers():
    text = "Authorization: Bearer abcdefghijklmnop123456"
    assert "abcdefg" not in mask(text)


def test_json_formatter_is_valid_json():
    record = logging.LogRecord("test", logging.INFO, __file__, 1, "问题 1 条", None, None)
    payload = json.loads(JsonFormatter().format(record))
    assert payload["level"] == "INFO"
    assert payload["msg"] == "问题 1 条"


def test_rate_limiter_allows_then_blocks():
    limiter = RateLimiter(2)
    assert limiter.hit("a")[0] is True
    assert limiter.hit("a")[0] is True
    allowed, left = limiter.hit("a")
    assert allowed is False and left == 0
    assert limiter.hit("b")[0] is True  # per-client buckets


def test_api_key_is_optional_when_unset(monkeypatch):
    monkeypatch.setattr(settings, "app_api_key", "")
    assert require_key(None) is None


def test_api_key_is_enforced_when_set(monkeypatch):
    monkeypatch.setattr(settings, "app_api_key", "secret")
    assert require_key("secret") is None
    with pytest.raises(HTTPException) as exc:
        require_key("wrong")
    assert exc.value.status_code == 401
