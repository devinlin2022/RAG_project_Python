"""OpenAI-compatible chat client.

Non-streaming in V1; V4 adds SSE without changing the signature.
"""

import logging

import requests

from rag.config import settings

log = logging.getLogger(__name__)


class LLMError(RuntimeError):
    """Raised with a message safe to show in the UI."""


def chat(
    messages: list[dict],
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> tuple[str, dict]:
    """Return (answer, usage). Raises LLMError on failure."""
    if not settings.llm_configured:
        raise LLMError("未配置 LLM_API_KEY，请在 .env 中填写后重启服务")

    payload = {
        "model": settings.llm_model,
        "messages": messages,
        "stream": False,
        "temperature": settings.temperature if temperature is None else temperature,
        "max_tokens": settings.max_tokens if max_tokens is None else max_tokens,
    }
    headers = {
        "Authorization": f"Bearer {settings.llm_api_key.strip()}",
        "Content-Type": "application/json; charset=utf-8",
    }

    try:
        response = requests.post(
            settings.chat_completions_url,
            json=payload,
            headers=headers,
            timeout=settings.llm_timeout,
        )
    except requests.exceptions.RequestException as exc:
        raise LLMError(f"网络请求失败：{exc}") from exc

    if response.status_code >= 400:
        detail = response.text.strip().replace("\n", " ")[:200]
        if response.status_code == 401:
            raise LLMError("API Key 无效（401），请检查 LLM_API_KEY")
        if response.status_code == 404:
            raise LLMError(
                f"接口或模型不存在（404）：{detail}。"
                "请核对 LLM_BASE_URL 是否以 /v1 结尾，以及 LLM_MODEL 是否为平台上的完整模型 ID"
            )
        raise LLMError(f"HTTP {response.status_code}：{detail}")

    try:
        data = response.json()
    except ValueError as exc:
        raise LLMError("响应不是合法 JSON，请检查 LLM_BASE_URL") from exc

    choices = data.get("choices") or []
    if not choices:
        raise LLMError(f"响应缺少 choices 字段：{str(data)[:200]}")

    message = choices[0].get("message") or {}
    content = message.get("content") or ""
    if isinstance(content, list):  # some providers return content parts
        content = "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )

    return content.strip() or "（模型返回空内容）", data.get("usage") or {}
