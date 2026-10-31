"""Explain a failed API call: wrong site, missing model, missing scope."""

from __future__ import annotations

import logging

import requests

log = logging.getLogger(__name__)

SITES = {
    "cn": "https://api.siliconflow.cn/v1",
    "com": "https://api.siliconflow.com/v1",
}
_TIMEOUT = 8
_cache: dict[tuple[str, str], str] = {}


def site_of(base_url: str) -> str:
    host = (base_url or "").lower()
    for name in SITES:
        if f"siliconflow.{name}" in host:
            return name
    return ""


def other_site(base_url: str) -> str:
    name = site_of(base_url)
    return SITES.get("com" if name == "cn" else "cn", "")


def _get(url: str, key: str):
    try:
        return requests.get(
            url,
            headers={"Authorization": f"Bearer {key}"},
            timeout=_TIMEOUT,
        )
    except requests.RequestException as exc:
        log.debug("diagnose probe failed: %s", exc)
        return None


def key_site_mismatch(base_url: str, key: str) -> str:
    if not key.strip():
        return ""
    if not site_of(base_url):
        return ""
    other = other_site(base_url)
    if not other:
        return ""

    cache_key = (other, key.strip()[-4:])
    if cache_key in _cache:
        return _cache[cache_key]

    hint = ""
    response = _get(other + "/models", key)
    if response is not None:
        if response.status_code == 200:
            this = site_of(base_url) or "当前地址"
            hint = (
                f"这把 Key 属于 api.siliconflow.{site_of(other)}，"
                f"而 SILICONFLOW_BASE_URL 指向 {this}。两个站点的 Key 不通用："
                f"把地址改成 {other} 就能用（模型 ID 也要换成那边的）。"
            )
        elif response.status_code == 402:
            hint = (
                f"这把 Key 在 api.siliconflow.{site_of(other)} 上能通过验证，"
                f"但那个账户余额不足（402）。当前地址指向 {site_of(base_url)}，Key 不通用。"
            )
    _cache[cache_key] = hint
    return hint


def model_alternatives(base_url: str, key: str, kind: str, limit: int = 4) -> list[str]:
    cache_key = (f"alts:{kind}", base_url)
    if cache_key in _cache:
        cached = _cache[cache_key]
        return cached.split("|") if cached else []

    response = _get(base_url.rstrip("/") + "/models", key)
    if response is None or response.status_code != 200:
        _cache[cache_key] = ""
        return []

    try:
        ids = [str(item.get("id") or "") for item in response.json().get("data", [])]
    except Exception:
        _cache[cache_key] = ""
        return []

    def belongs(model_id: str) -> bool:
        low = model_id.lower()
        if kind == "embed":
            return "embed" in low
        if kind == "rerank":
            return "rerank" in low
        if kind == "vision":
            return "vl" in low or "vision" in low
        return "vl" not in low and "embed" not in low and "rerank" not in low

    picked = [model_id for model_id in ids if model_id and belongs(model_id)][:limit]
    _cache[cache_key] = "|".join(picked)
    return picked


def explain_401(base_url: str, key: str) -> str:
    mismatch = key_site_mismatch(base_url, key)
    if mismatch:
        return mismatch
    return (
        "这把 Key 在当前站点上通不过验证，也没有在另一个站点上通过——"
        "多半已被删除或没复制全，去控制台重新生成一把。"
    )


def explain_404(base_url: str, key: str, wanted: str, kind: str) -> str:
    alternatives = model_alternatives(base_url, key, kind)
    if not alternatives:
        return f"这个站点上没有 {wanted}，也没查到可替换的同类模型。"
    return f"这个站点上没有 {wanted}，可用的同类模型：" + "、".join(alternatives)


def reset_cache() -> None:
    _cache.clear()
