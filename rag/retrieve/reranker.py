"""Cross-encoder reranking through the /v1/rerank endpoint."""

import logging

import requests

from rag.config import settings
from rag.diagnose import explain_401, explain_404
from rag.models import is_model_error
from rag.net import describe_status, post_json

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

    headers = {
        "Authorization": f"Bearer {settings.llm_api_key.strip()}",
        "Content-Type": "application/json",
    }
    chain = settings.rerank_chain

    for index, model in enumerate(chain):
        payload = {
            "model": model,
            "query": query,
            "documents": [doc[:MAX_DOC_CHARS] for doc in documents],
            "top_n": top_n or len(documents),
            "return_documents": False,
        }
        try:
            response = post_json(
                settings.rerank_url,
                headers,
                payload,
                settings.rerank_timeout,
                min_interval=settings.http_min_interval,
                retries=settings.http_retries,
            )
        except requests.RequestException as exc:
            raise RerankError(f"精排请求失败：{exc}") from exc

        if is_model_error(response.status_code, response.text) and index < len(chain) - 1:
            log.warning("rerank model %s unavailable, trying %s", model, chain[index + 1])
            continue

        if response.status_code == 404:
            raise RerankError(
                f"精排模型不存在（404）：{model}。"
                + explain_404(settings.llm_url, settings.llm_api_key, model, "rerank")
                + "也可以设 RERANK_ENABLED=false 先关掉精排，"
                "或用 RERANK_FALLBACKS 换一个可用模型。"
            )
        if response.status_code in (401, 402, 403):
            explain = (
                explain_401(settings.llm_url, settings.llm_api_key)
                if response.status_code == 401
                else f"当前站点 {settings.llm_url}"
            )
            raise RerankError(
                f"精排不可用：{describe_status(response.status_code)}，{response.text[:100]}。"
                f"当前 Key 尾号 {settings.llm_api_key.strip()[-4:] or '空'}。{explain}"
            )
        if response.status_code == 429:
            raise RerankError(
                "精排触发限流（429）：问答、向量、精排共用一个 Key，把 HTTP_MIN_INTERVAL 调大"
                f"（当前 {settings.http_min_interval}s）或减少候选数（当前 {len(documents)} 条）"
            )
        if response.status_code >= 400:
            raise RerankError(f"精排返回 {response.status_code}：{response.text[:160]}")

        try:
            results = response.json()["results"]
            pairs = [(int(item["index"]), float(item["relevance_score"])) for item in results]
        except (KeyError, TypeError, ValueError) as exc:
            raise RerankError(f"精排响应无法解析：{response.text[:160]}") from exc

        return sorted(pairs, key=lambda pair: -pair[1])

    raise RerankError("精排没有可用模型，请检查 RERANK_MODEL 与 RERANK_FALLBACKS")
