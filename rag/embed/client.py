"""Embedding client. bge-m3 on SiliconFlow is free and needs no local torch.

Used only to find semantic breakpoints and, from V3, to build the vector index.
"""

import logging
import time

from rag.config import settings

log = logging.getLogger(__name__)

# A bad key or an empty balance must not cost a failed round trip for every
# section that needs a breakpoint. After an auth or billing failure the client
# stops trying for a while and callers fall back to character splitting.
_COOLDOWN_SECONDS = 300
_disabled_until = 0.0


class EmbedError(Exception):
    """Embedding request failed."""


def _trip(reason: str) -> None:
    global _disabled_until
    _disabled_until = time.time() + _COOLDOWN_SECONDS
    log.warning("embedding paused for %ss: %s", _COOLDOWN_SECONDS, reason)


def embed_ready() -> bool:
    return settings.embed_ready and time.time() >= _disabled_until


def embed_texts(texts: list[str], batch: int | None = None) -> list[list[float]]:
    """Embed texts in batches, preserving order."""
    import requests

    if not texts:
        return []
    api_key = settings.embed_api_key_resolved
    if not api_key:
        raise EmbedError("未配置 Embedding API Key（见 .env 的 LLM_API_KEY 或 EMBED_API_KEY）")
    if time.time() < _disabled_until:
        raise EmbedError("Embedding 已暂停（近期出现认证或余额错误），改按字符切分")

    batch = batch or settings.embed_batch
    url = settings.embed_base_url_resolved + "/embeddings"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    vectors: list[list[float]] = []

    for start in range(0, len(texts), batch):
        window = texts[start : start + batch]
        payload = {"model": settings.embed_model, "input": window, "encoding_format": "float"}
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=settings.llm_timeout)
        except Exception as exc:
            raise EmbedError(f"Embedding 请求失败：{exc}") from exc

        if response.status_code in (401, 402):
            _trip(f"HTTP {response.status_code}")
            raise EmbedError(
                f"Embedding 不可用（{response.status_code}）：{response.text[:120]}"
            )
        if response.status_code == 404:
            _trip("model not found")
            raise EmbedError(f"Embedding 模型不存在（404）：{settings.embed_model}")
        if response.status_code >= 400:
            raise EmbedError(f"Embedding 返回 {response.status_code}：{response.text[:200]}")

        try:
            body = response.json()
            items = sorted(body["data"], key=lambda item: item.get("index", 0))
            vectors.extend([item["embedding"] for item in items])
        except Exception as exc:
            raise EmbedError(f"Embedding 响应无法解析：{response.text[:200]}") from exc

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


def timed_embed(texts: list[str]) -> tuple[list[list[float]], int]:
    started = time.perf_counter()
    vectors = embed_texts(texts)
    return vectors, int((time.perf_counter() - started) * 1000)
