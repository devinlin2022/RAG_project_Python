"""Verify .env against the live API: model ids, chat, embedding, rerank, OCR.

Usage:
    python check_llm.py            # chat + embedding + rerank
    python check_llm.py --ocr      # also one vision call (costs a page)
"""

import sys
import time

import requests

from rag.config import settings


def mask(key: str) -> str:
    key = key.strip()
    return f"{key[:6]}…{key[-4:]}" if len(key) > 12 else "已填写"


def show_config() -> None:
    print("== 当前配置 ==")
    print(f"LLM_BASE_URL : {settings.llm_base_url}")
    print(f"LLM_MODEL    : {settings.llm_model}  (问答)")
    print(f"OCR_MODEL    : {settings.ocr_model}  (OCR, {'已启用' if settings.ocr_enabled else '未启用'})")
    print(f"VISION_MODEL : {settings.vision_model}  (图注)")
    print(f"EMBED_MODEL  : {settings.embed_model}  (向量)")
    print(f"RERANK_MODEL : {settings.rerank_model}  (精排, {'已启用' if settings.rerank_enabled else '关闭'})")
    print(f"LLM_API_KEY  : {mask(settings.llm_api_key) if settings.llm_configured else '未配置'}")


def check_models(headers: dict) -> tuple[bool, list[str]]:
    try:
        resp = requests.get(settings.models_url, headers=headers, timeout=20)
    except requests.exceptions.RequestException as exc:
        print(f"\n无法连接 {settings.models_url}\n  {exc}")
        print("若开启了代理/VPN，请关闭，或将 api.siliconflow.cn 加入 NO_PROXY。")
        return False, []

    if resp.status_code == 401:
        print("\nAPI Key 无效（401）。注意 .cn 与 .com 是两个站点，Key 与模型目录都不通用。")
        return False, []
    if resp.status_code != 200:
        print(f"\n获取模型列表失败：HTTP {resp.status_code} {resp.text[:200]}")
        return False, []

    ids = [item.get("id", "") for item in resp.json().get("data", [])]
    wanted: dict[str, list[str]] = {}
    for model_id, role in (
        (settings.llm_model, "问答"),
        (settings.ocr_model, "OCR"),
        (settings.vision_model, "图注"),
        (settings.embed_model, "向量"),
        (settings.rerank_model, "精排"),
    ):
        wanted.setdefault(model_id, []).append(role)

    print(f"\n== 模型可用性 ==\n平台可用模型 {len(ids)} 个")
    for model_id, roles in wanted.items():
        print(f"  {'OK  ' if model_id in ids else 'MISS'} {model_id:38s} ({'/'.join(roles)})")

    missing = [m for m in wanted if m not in ids]
    if missing:
        print("\n以上带 MISS 的模型 ID 在当前站点不存在，请到模型广场复制完整 ID。")
    return not missing, missing


def check_chat(headers: dict) -> bool:
    started = time.perf_counter()
    try:
        resp = requests.post(
            settings.chat_completions_url,
            json={
                "model": settings.llm_model,
                "messages": [{"role": "user", "content": "只回复 pong"}],
                "max_tokens": 8,
                "temperature": 0,
            },
            headers=headers,
            timeout=settings.llm_timeout,
        )
    except requests.exceptions.RequestException as exc:
        print(f"  FAIL chat      {exc}")
        return False

    elapsed = time.perf_counter() - started
    if resp.status_code >= 400:
        print(f"  FAIL chat      HTTP {resp.status_code} {resp.text[:160]}")
        if resp.status_code == 402:
            print("                 账户余额不足，请先充值")
        return False

    content = ((resp.json().get("choices") or [{}])[0].get("message") or {}).get("content", "")
    print(f"  OK   chat      {elapsed:.1f}s  返回 {content.strip()[:20]!r}")
    return True


def check_embedding(headers: dict) -> bool:
    started = time.perf_counter()
    resp = requests.post(
        settings.embed_base_url_resolved + "/embeddings",
        json={"model": settings.embed_model, "input": "连通性测试", "encoding_format": "float"},
        headers=headers,
        timeout=settings.llm_timeout,
    )
    elapsed = time.perf_counter() - started
    if resp.status_code >= 400:
        print(f"  FAIL embed     HTTP {resp.status_code} {resp.text[:160]}")
        return False
    dim = len(resp.json()["data"][0]["embedding"])
    print(f"  OK   embed     {elapsed:.1f}s  {dim} 维")
    return True


def check_rerank(headers: dict) -> bool:
    """One rerank call: proves both the endpoint and the model id."""
    if not settings.rerank_enabled:
        print("  SKIP rerank    已关闭（RERANK_ENABLED=false）")
        return True
    from rag.retrieve.reranker import RerankError, rerank

    started = time.perf_counter()
    try:
        pairs = rerank("付款账期", ["验收合格后三十日内付款", "本页描述采购流程示意图"])
    except RerankError as exc:
        print(f"  FAIL rerank    {exc}")
        return False
    elapsed = time.perf_counter() - started
    top = pairs[0] if pairs else (None, 0.0)
    print(f"  OK   rerank    {elapsed:.1f}s  首位得分 {top[1]:.3f}  {settings.rerank_model}")
    return top[0] == 0


def check_ocr() -> bool:
    if not settings.ocr_enabled:
        print("  SKIP ocr       未启用（OCR_ENABLED=false），表格通道不受影响")
        return True
    from rag.ingest.ocr import OcrError, ocr_page

    png = _probe_png()
    try:
        result = ocr_page(png)
    except (OcrError, ValueError) as exc:
        print(f"  FAIL ocr       {exc}")
        return False
    print(f"  OK   ocr       {result.ms / 1000:.1f}s  {len(result.text)} 字  {settings.ocr_model}")
    return True


def _probe_png() -> bytes:
    """Render a white page with one line of text, so the call is cheap."""
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (600, 200), "white")
    ImageDraw.Draw(image).text((30, 80), "RAG DEMO 12345", fill="black")
    import io

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def main() -> int:
    show_config()
    if not settings.llm_configured:
        print("\n请在项目根目录创建 .env（cp .env.example .env），填入 LLM_API_KEY 后重试。")
        return 1

    headers = {
        "Authorization": f"Bearer {settings.llm_api_key.strip()}",
        "Content-Type": "application/json",
    }

    models_ok, _ = check_models(headers)
    print("\n== 连通性 ==")
    chat_ok = check_chat(headers)
    embed_ok = check_embedding(headers)
    rerank_ok = check_rerank(headers)
    ocr_ok = check_ocr() if "--ocr" in sys.argv else True
    if "--ocr" not in sys.argv and settings.ocr_enabled:
        print("  SKIP ocr       加 --ocr 可实测一次（会产生一次调用）")

    if not (chat_ok and embed_ok and ocr_ok and rerank_ok):
        print("\n有通道不可用，先修好再启动。")
        return 2
    if not models_ok:
        print("\n问答可用，但上表有模型 ID 不存在（对应通道会在运行时降级或报错）。")
        return 3

    print("\n全部通过，可以启动：python run.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
