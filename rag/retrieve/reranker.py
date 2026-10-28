"""Cross-encoder reranking through SiliconFlow's /v1/rerank endpoint.

A hosted cross-encoder replaces a local bge-reranker load: no torch, no 2 GB
download, and the same model (`bge-reranker-v2-m3`). The call is one request per
query, so latency is bounded and visible in the UI.
"""

import logging

import requests

from rag.config import settings
from rag.net import post_json

log = logging.getLogger(__name__)

# Cross-encoders truncate anyway; sending 4000 chars per doc only inflates the bill.
MAX_DOC_CHARS = 2000


class RerankError(RuntimeError):
    """Message is safe to show in the UI."""


def ready() -> bool:
    return settings.rerank_ready


def rerank(query: str, documents: list[str], top_n: int | None = None) -> list[tuple[int, float]]:
    """Return [(document index, relevance score)] sorted by score."""
    if not documents:
        return []
    if not settings.rerank_ready:
        raise RerankError("未启用精排（RERANK_ENABLED=false 或未配置 Key）")

    payload = {
        "model": settings.rerank_model,
        "query": query,
        "documents": [doc[:MAX_DOC_CHARS] for doc in documents],
        "top_n": top_n or len(documents),
        "return_documents": False,
    }
    headers = {
        "Authorization": f"Bearer {settings.llm_api_key.strip()}",
        "Content-Type": "application/json",
    }
    try:
        response = post_json(
            settings.rerank_url, headers, payload, settings.rerank_timeout, retries=1
        )
    except requests.RequestException as exc:
        raise RerankError(f"精排请求失败：{exc}") from exc

    if response.status_code == 404:
        raise RerankError(
            f"精排模型不存在（404）：{settings.rerank_model}。"
            "请在平台模型广场核对 ID，或设置 RERANK_ENABLED=false"
        )
    if response.status_code in (401, 402):
        raise RerankError(f"精排不可用（{response.status_code}）：{response.text[:120]}")
    if response.status_code >= 400:
        raise RerankError(f"精排返回 {response.status_code}：{response.text[:160]}")

    try:
        results = response.json()["results"]
        pairs = [(int(item["index"]), float(item["relevance_score"])) for item in results]
    except (KeyError, TypeError, ValueError) as exc:
        raise RerankError(f"精排响应无法解析：{response.text[:160]}") from exc

    return sorted(pairs, key=lambda pair: -pair[1])
