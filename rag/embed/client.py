"""Embedding client. Vector index, semantic breakpoints, one HTTP shape."""

import logging
import time
from concurrent.futures import ThreadPoolExecutor

import requests

from rag.config import settings
from rag.net import post_json

log = logging.getLogger(__name__)

# After an auth or billing failure, stop retrying for a while: otherwise every
# section that needs a breakpoint pays a failed round trip.
_COOLDOWN_SECONDS = 300
_disabled_until = 0.0

# Last observed usage, for the token panel. {"prompt_tokens": n, "requests": k}
last_usage: dict = {}


class EmbedError(Exception):
    """Embedding request failed."""


def _trip(reason: str) -> None:
    global _disabled_until
    _disabled_until = time.time() + _COOLDOWN_SECONDS
    log.warning("embedding paused for %ss: %s", _COOLDOWN_SECONDS, reason)


def embed_ready() -> bool:
    return settings.embed_ready and time.time() >= _disabled_until


def _batches(texts: list[str], size: int) -> list[list[str]]:
    return [texts[start : start + size] for start in range(0, len(texts), size)]


def _embed_batch(items: list[str]) -> tuple[list[list[float]], int]:
    """One HTTP call. Returns (vectors, prompt_tokens)."""
    api_key = settings.embed_api_key_resolved
    payload = {
        "model": settings.embed_model,
        "input": items,
        "encoding_format": "float",
    }
    try:
        response = post_json(
            settings.embed_url,
            {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            payload,
            settings.llm_timeout,
            min_interval=settings.ocr_min_interval,
            retries=settings.http_retries,
        )
    except requests.Timeout as exc:
        raise EmbedError(f"Embedding 超时（{settings.llm_timeout}s）") from exc
    except Exception as exc:
        raise EmbedError(f"Embedding 请求失败：{exc}") from exc

    if response.status_code in (401, 402):
        _trip(f"HTTP {response.status_code}")
        raise EmbedError(f"Embedding 不可用（{response.status_code}）：{response.text[:120]}")
    if response.status_code == 429:
        raise EmbedError("Embedding 触发限流（429），请调小 EMBED_BATCH 或 EMBED_CONCURRENCY")
    if response.status_code == 404:
        _trip("model not found")
        raise EmbedError(f"Embedding 模型不存在（404）：{settings.embed_model}")
    if response.status_code >= 400:
        raise EmbedError(f"Embedding 返回 {response.status_code}：{response.text[:200]}")

    try:
        body = response.json()
        items_out = sorted(body["data"], key=lambda item: item.get("index", 0))
        vectors = [item["embedding"] for item in items_out]
    except Exception as exc:
        raise EmbedError(f"Embedding 响应无法解析：{response.text[:200]}") from exc

    tokens = int((body.get("usage") or {}).get("prompt_tokens") or 0)
    return vectors, tokens


def embed_texts(texts: list[str], batch: int | None = None) -> list[list[float]]:
    """Embed any number of texts, in parallel batches, order preserved."""
    if not texts:
        return []
    api_key = settings.embed_api_key_resolved
    if not api_key:
        raise EmbedError("未配置 API Key，请在 .env 填写 SILICONFLOW_API_KEY")
    if time.time() < _disabled_until:
        raise EmbedError("Embedding 已暂停（近期出现认证或余额错误），改按字符切分")

    size = batch or settings.embed_batch
    groups = _batches(texts, size)
    workers = max(1, min(int(settings.embed_concurrency), len(groups)))

    vectors: list[list[float]] = []
    tokens = 0
    if workers == 1:
        for group in groups:
            chunk, used = _embed_batch(group)
            vectors.extend(chunk)
            tokens += used
    else:
        # Order matters: results are reassembled positionally, so each future
        # keeps its index instead of the completion order.
        results: list[tuple[list[list[float]], int]] = [([], 0)] * len(groups)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_embed_batch, group): i for i, group in enumerate(groups)}
            for future, index in futures.items():
                results[index] = future.result()
        for chunk, used in results:
            vectors.extend(chunk)
            tokens += used

    last_usage.clear()
    last_usage.update(
        prompt_tokens=tokens, requests=len(groups), model=settings.embed_model
    )

    if len(vectors) != len(texts):
        raise EmbedError(f"向量数量不一致：期望 {len(texts)}，实际 {len(vectors)}")
    return vectors


def cosine(a: list[float], b: list[float]) -> float:
    import numpy as np

    va = np.asarray(a, dtype="float32")
    vb = np.asarray(b, dtype="float32")
    denominator = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if denominator == 0:
        return 0.0
    return float(np.dot(va, vb) / denominator)
