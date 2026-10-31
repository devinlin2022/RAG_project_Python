"""OpenAI-compatible chat client. Non-streaming and SSE streaming."""

import json
import logging
from typing import Iterator

import requests

from rag.config import settings
from rag.net import post_json

log = logging.getLogger(__name__)


class LLMError(RuntimeError):
    """Raised with a message safe to show in the UI."""


def _headers() -> dict:
    if not settings.llm_configured:
        raise LLMError("未配置 SILICONFLOW_API_KEY：请在 .env 填写后重启，或在界面「运行状态」里临时粘贴")
    return {
        "Authorization": f"Bearer {settings.llm_api_key.strip()}",
        "Content-Type": "application/json; charset=utf-8",
    }


def _explain(response) -> LLMError:
    detail = response.text.strip().replace("\n", " ")[:200]
    if response.status_code == 401:
        from rag.diagnose import explain_401

        return LLMError(
            f"API Key 被拒绝（401）：{detail or '平台未说明原因'}。"
            f"当前 key 尾号 {settings.llm_api_key.strip()[-4:] or '空'}、"
            f"地址 {settings.llm_url}、模型 {settings.llm_model}。"
            f"{explain_401(settings.llm_url, settings.llm_api_key)}"
            "问答、向量、精排、视觉共用 SILICONFLOW_API_KEY，改一处即可。"
        )
    if response.status_code == 402:
        return LLMError(
            f"账户余额不足（402）：{detail}。"
            f"当前用的是 {settings.llm_url}，充值要在同一个站点充；"
            "另一个站点的余额不会共享。"
        )
    if response.status_code == 429:
        return LLMError(
            "触发限流（429）：并发请求过多，同一把 Key 的问答、向量、精排共用配额。"
            f"把 HTTP_MIN_INTERVAL 调大（当前 {settings.http_min_interval}s）或稍等几秒重试"
        )
    if response.status_code == 404:
        from rag.diagnose import explain_404

        return LLMError(
            f"接口或模型不存在（404）：{detail}。"
            + explain_404(settings.llm_url, settings.llm_api_key, settings.llm_model, "chat")
            + "另外确认 SILICONFLOW_BASE_URL 以 /v1 结尾。"
        )
    return LLMError(f"HTTP {response.status_code}：{detail}")


def chat(
    messages: list[dict],
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> tuple[str, dict]:
    """Return (answer, usage). Raises LLMError on failure."""
    headers = _headers()
    payload = {
        "model": settings.llm_model,
        "messages": messages,
        "stream": False,
        "temperature": settings.temperature if temperature is None else temperature,
        "max_tokens": settings.max_tokens if max_tokens is None else max_tokens,
    }

    try:
        response = post_json(
            settings.chat_completions_url,
            headers,
            payload,
            settings.llm_timeout,
            min_interval=settings.http_min_interval,
            retries=settings.http_retries,
        )
    except requests.exceptions.RequestException as exc:
        raise LLMError(f"网络请求失败：{exc}") from exc

    if response.status_code >= 400:
        raise _explain(response)

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


def _post_stream(payload: dict, headers: dict):
    # stream=True only reads headers, so a retry here never replays a partial body.
    return post_json(
        settings.chat_completions_url,
        headers,
        payload,
        settings.llm_timeout,
        min_interval=settings.http_min_interval,
        retries=settings.http_retries,
        stream=True,
    )


def chat_stream(
    messages: list[dict],
    temperature: float | None = None,
    max_tokens: int | None = None,
    on_usage=None,
) -> Iterator[str]:
    """Yield content deltas as they arrive. Raises LLMError before the first token."""
    headers = _headers()
    payload = {
        "model": settings.llm_model,
        "messages": messages,
        "stream": True,
        "temperature": settings.temperature if temperature is None else temperature,
        "max_tokens": settings.max_tokens if max_tokens is None else max_tokens,
        # Without this a streamed answer reports no token usage at all, and the
        # UI would have nothing to show in the cost panel.
        "stream_options": {"include_usage": True},
    }

    try:
        response = _post_stream(payload, headers)
    except requests.exceptions.RequestException as exc:
        raise LLMError(f"网络请求失败：{exc}") from exc

    if response.status_code == 400:
        # Some OpenAI-compatible gateways reject stream_options outright.
        payload.pop("stream_options", None)
        try:
            response = _post_stream(payload, headers)
        except requests.exceptions.RequestException as exc:
            raise LLMError(f"网络请求失败：{exc}") from exc

    if response.status_code >= 400:
        raise _explain(response)

    produced = False
    # Decode explicitly: an SSE response without a charset header makes requests
    # fall back to ISO-8859-1, which turns every Chinese character into mojibake.
    for raw in response.iter_lines(decode_unicode=False):
        if not raw:
            continue
        try:
            line = raw.decode("utf-8")
        except UnicodeDecodeError:
            line = raw.decode("utf-8", errors="replace")
        if not line.startswith("data:"):
            continue
        chunk = line[5:].strip()
        if chunk == "[DONE]":
            break
        try:
            data = json.loads(chunk)
        except ValueError:
            continue
        if data.get("usage") and on_usage:
            on_usage(data["usage"])
        choices = data.get("choices") or []
        if not choices:
            continue
        delta = choices[0].get("delta") or {}
        piece = delta.get("content")
        if piece:
            produced = True
            yield piece

    if not produced:
        yield ""
