"""Verify .env: base URL, key, model, then send one tiny request.

Usage: python check_llm.py
"""

import sys

import requests

from rag.config import settings


def mask(key: str) -> str:
    key = key.strip()
    return f"{key[:6]}…{key[-4:]}" if len(key) > 12 else "已填写"


def main() -> int:
    print("== 当前配置 ==")
    print(f"LLM_BASE_URL : {settings.llm_base_url}")
    print(f"LLM_MODEL    : {settings.llm_model}")
    print(f"LLM_API_KEY  : {mask(settings.llm_api_key) if settings.llm_configured else '未配置'}")

    if not settings.llm_configured:
        print("\n请在项目根目录创建 .env（cp .env.example .env），填入 LLM_API_KEY 后重试。")
        return 1

    headers = {
        "Authorization": f"Bearer {settings.llm_api_key.strip()}",
        "Content-Type": "application/json",
    }

    # 1) list models
    try:
        resp = requests.get(settings.models_url, headers=headers, timeout=20)
    except requests.exceptions.RequestException as exc:
        print(f"\n无法连接 {settings.models_url}")
        print(f"  {exc}")
        print("若开启了代理/VPN，请关闭，或将 api.siliconflow.cn 加入 NO_PROXY。")
        return 2

    if resp.status_code == 401:
        print("\nAPI Key 无效（401），请重新复制完整密钥（以 sk- 开头）。")
        return 2

    if resp.status_code != 200:
        print(f"\n获取模型列表失败：HTTP {resp.status_code} {resp.text[:200]}")
    else:
        ids = [item.get("id", "") for item in resp.json().get("data", [])]
        deepseek = sorted(i for i in ids if "deepseek" in i.lower())
        print(f"\n平台可用模型 {len(ids)} 个，其中 DeepSeek 相关 {len(deepseek)} 个：")
        for model_id in deepseek:
            mark = "  ← 当前配置" if model_id == settings.llm_model else ""
            print(f"  - {model_id}{mark}")
        if ids and settings.llm_model not in ids:
            print(f"\n注意：LLM_MODEL={settings.llm_model} 不在列表中，请从上表复制完整 ID。")

    # 2) minimal chat request
    payload = {
        "model": settings.llm_model,
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 8,
        "temperature": 0,
    }
    try:
        resp = requests.post(
            settings.chat_completions_url,
            json=payload,
            headers=headers,
            timeout=settings.llm_timeout,
        )
    except requests.exceptions.RequestException as exc:
        print(f"\n对话请求失败：{exc}")
        return 2

    if resp.status_code >= 400:
        print(f"\n对话请求返回 HTTP {resp.status_code}")
        print(f"  {resp.text[:300]}")
        if resp.status_code == 404:
            print("  通常表示模型 ID 不存在，或 LLM_BASE_URL 末尾缺少 /v1。")
        return 2

    data = resp.json()
    content = (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
    print(f"\n对话测试成功，返回：{content.strip()[:40]!r}")
    print("配置可用，可以启动：python run.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
