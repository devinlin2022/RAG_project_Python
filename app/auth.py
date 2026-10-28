"""API key auth, per-client rate limiting, and JSON logs with secret masking.

All three are opt-in: with an empty APP_API_KEY / UI_PASSWORD the service runs
open for local use, which is what a demo needs. Setting them is one .env line.
"""

import json
import logging
import re
import threading
import time
from collections import deque

from fastapi import Header, HTTPException, Request

from rag.config import settings

_SECRET = re.compile(r"sk-[A-Za-z0-9]{10,}")
_BEARER = re.compile(r"Bearer\s+[A-Za-z0-9\-._~+/]{10,}=*")


def mask(text: str) -> str:
    """Remove anything that looks like a credential before it reaches a log."""
    if not text:
        return text
    text = _SECRET.sub("sk-***", text)
    return _BEARER.sub("Bearer ***", text)


class RedactionFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:
            return True
        cleaned = mask(message)
        if cleaned != message:
            record.msg = cleaned
            record.args = ()
        return True


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def setup_logging(level: int = logging.INFO) -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter() if settings.log_json else logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s - %(message)s"
    ))
    handler.addFilter(RedactionFilter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)
    for noisy in ("httpx", "httpcore", "uvicorn.access"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


class RateLimiter:
    """Sliding 60-second window per client."""

    def __init__(self, limit_per_min: int) -> None:
        self.limit = max(int(limit_per_min), 1)
        self._events: dict[str, deque] = {}
        self._lock = threading.Lock()

    def hit(self, key: str) -> tuple[bool, int]:
        now = time.time()
        with self._lock:
            bucket = self._events.setdefault(key, deque())
            while bucket and now - bucket[0] > 60:
                bucket.popleft()
            if len(bucket) >= self.limit:
                return False, self.limit - len(bucket)
            bucket.append(now)
            return True, self.limit - len(bucket)


LIMITER = RateLimiter(settings.rate_limit_per_min)


def _client(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def require_key(x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> None:
    """No-op when APP_API_KEY is empty."""
    expected = settings.app_api_key.strip()
    if not expected:
        return
    if (x_api_key or "").strip() != expected:
        raise HTTPException(status_code=401, detail="缺少或错误的 X-API-Key")


def rate_limit(request: Request) -> None:
    allowed, left = LIMITER.hit(_client(request))
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail=f"请求过于频繁（上限 {LIMITER.limit}/分钟），请稍后再试",
            headers={"Retry-After": "60"},
        )
