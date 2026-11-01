"""Verify .env against the live API: model ids, chat, embedding, rerank, vision.

Usage:
    python check_llm.py                # models + chat + embedding + rerank
    python check_llm.py --ocr          # also one vision call
    python check_llm.py --candidates   # every fallback id, one call each
    python check_llm.py --models       # print what this key can use
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
    print(f"SILICONFLOW_BASE_URL : {settings.llm_base_url}")
    print(f"SILICONFLOW_API_KEY  : {mask(settings.llm_api_key) if settings.llm_configured else '未配置'}")
    print(f"LLM_MODEL            : {settings.llm_model}  (问答，备选 {len(settings.llm_chain) - 1})")
    print(f"EMBED_MODEL          : {settings.embed_model}  (向量，候选 {len(settings.embed_chain) - 1})")
    print(f"RERANK_MODEL         : {settings.rerank_model}  (精排, "
          f"{'启用' if settings.rerank_enabled else '关闭'}，备选 {len(settings.rerank_chain) - 1})")
    print(f"OCR_MODEL            : {settings.ocr_model}  (扫描页转录, {settings.vision_state[1]}，"
          f"备选 {len(settings.ocr_chain) - 1})")
    print(f"VISION_MODEL         : {settings.vision_model}  (图表描述，备选 {len(settings.vision_chain) - 1})")


def fetch_models(headers: dict) -> list[str] | None:
    try:
        resp = requests.get(settings.models_url, headers=headers, timeout=20)
    except requests.exceptions.RequestException as exc:
        print(f"\n无法连接 {settings.models_url}\n  {exc}")
        print("若开启了代理/VPN，请关闭，或将 api.siliconflow.cn 加入 NO_PROXY。")
        return None

    if resp.status_code == 401:
        print("\nAPI Key 无效（401）。注意 .cn 与 .com 是两个站点，Key 与模型目录都不通用。")
        return None
    if resp.status_code != 200:
        print(f"\n获取模型列表失败：HTTP {resp.status_code} {resp.text[:200]}")
        return None
    return [item.get("id", "") for item in resp.json().get("data", [])]


def check_models(ids: list[str]) -> tuple[bool, list[str]]:
    """Primary ids decide pass/fail; a missing fallback is only reported."""
    groups = (
        ("问答", settings.llm_chain),
        ("向量", settings.embed_chain),
        ("精排", settings.rerank_chain),
        ("OCR", settings.ocr_chain),
        ("图表", settings.vision_chain),
    )
    print(f"\n== 模型可用性 ==\n平台可用模型 {len(ids)} 个")
    primary_missing: list[str] = []
    fallback_missing: list[str] = []
    for role, chain in groups:
        for position, model_id in enumerate(chain):
            present = model_id in ids
            tag = "OK  " if present else ("MISS" if position == 0 else "MISS-")
            print(f"  {tag} {model_id:44s} ({role}{'' if position == 0 else '备选'})")
            if not present:
                (primary_missing if position == 0 else fallback_missing).append(model_id)

    if primary_missing:
        print("\n主模型不存在，对应通道会直接失败。`--models` 可查看当前站点现货。")
    if fallback_missing:
        print(f"备选模型有 {len(fallback_missing)} 个不存在，运行时会被跳过，不影响主模型。")
    return not primary_missing, primary_missing


def print_catalog(ids: list[str]) -> None:
    """Group the catalogue so a wrong model id can be replaced in one glance."""
    groups = {
        "对话 / 推理": ("deepseek", "qwen", "glm", "kimi", "minimax"),
        "向量": ("embedding", "bge-m3", "gte"),
        "精排": ("reranker", "rerank"),
        "视觉": ("-vl", "vl-", "vision", "ocr"),
    }
    print(f"\n== 平台模型目录（{len(ids)} 个）==")
    for title, needles in groups.items():
        hits = [m for m in ids if any(n in m.lower() for n in needles)]
        if not hits:
            continue
        print(f"\n[{title}] {len(hits)} 个")
        for model_id in hits[:40]:
            print(f"  {model_id}")
        if len(hits) > 40:
            print(f"  … 另有 {len(hits) - 40} 个")


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
        settings.embed_url,
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
        print(f"  SKIP vision    {settings.vision_state[1]}；表格通道不受影响")
        return True
    from rag.ingest.ocr import OcrError, ocr_page

    png = _probe_png()
    try:
        result = ocr_page(png)
    except (OcrError, ValueError) as exc:
        print(f"  FAIL vision    {exc}")
        return False
    print(f"  OK   vision    {result.ms / 1000:.1f}s  {len(result.text)} 字  {settings.ocr_model}")
    return True


def check_candidates(headers: dict) -> bool:
    """One real call per configured id, so a dead fallback is known up front."""
    from rag.ingest.ocr import OCR_PROMPT, vision_chat

    print("\n== 候选模型实测 ==")
    ok = True

    for model_id in settings.llm_chain:
        started = time.perf_counter()
        try:
            resp = requests.post(
                settings.chat_completions_url,
                json={"model": model_id, "messages": [{"role": "user", "content": "只回复 ok"}],
                      "max_tokens": 8, "temperature": 0},
                headers=headers,
                timeout=settings.llm_timeout,
            )
            if resp.status_code >= 400:
                raise RuntimeError(f"HTTP {resp.status_code} {resp.text[:80]}")
            print(f"  OK   chat    {time.perf_counter() - started:5.1f}s  {model_id}")
        except Exception as exc:
            ok = False
            print(f"  FAIL chat    {model_id}  {exc}")

    base_dim = 0
    for model_id in settings.embed_chain:
        started = time.perf_counter()
        try:
            resp = requests.post(
                settings.embed_url,
                json={"model": model_id, "input": "连通性测试", "encoding_format": "float"},
                headers=headers,
                timeout=settings.llm_timeout,
            )
            if resp.status_code >= 400:
                raise RuntimeError(f"HTTP {resp.status_code} {resp.text[:80]}")
            dim = len(resp.json()["data"][0]["embedding"])
            base_dim = base_dim or dim
            note = "" if dim == base_dim else f"  ！维度与主模型 {base_dim} 不同，切换需重建索引"
            print(f"  OK   embed   {time.perf_counter() - started:5.1f}s  {dim} 维  {model_id}{note}")
        except Exception as exc:
            ok = False
            print(f"  FAIL embed   {model_id}  {exc}")

    for model_id in settings.rerank_chain:
        started = time.perf_counter()
        try:
            resp = requests.post(
                settings.rerank_url,
                json={"model": model_id, "query": "付款账期",
                      "documents": ["验收合格后三十日内付款", "本页描述采购流程"],
                      "top_n": 2},
                headers=headers,
                timeout=settings.rerank_timeout,
            )
            if resp.status_code >= 400:
                raise RuntimeError(f"HTTP {resp.status_code} {resp.text[:80]}")
            top = resp.json()["results"][0]
            print(f"  OK   rerank  {time.perf_counter() - started:5.1f}s  "
                  f"首位 {top['relevance_score']:.3f}  {model_id}")
        except Exception as exc:
            ok = False
            print(f"  FAIL rerank  {model_id}  {exc}")

    png = _probe_png()
    for model_id in settings.vision_chain:
        try:
            result = vision_chat(png, OCR_PROMPT, [model_id])
            got = " ".join(result.text.split())[:16]
            print(f"  OK   vision  {result.ms / 1000:5.1f}s  {got!r}  {model_id}")
        except Exception as exc:
            ok = False
            print(f"  FAIL vision  {model_id}  {exc}")

    return ok


def _probe_png() -> bytes:
    """Render one line of text on white, so the call is cheap."""
    import io

    from PIL import Image, ImageDraw

    image = Image.new("RGB", (600, 200), "white")
    ImageDraw.Draw(image).text((30, 80), "RAG DEMO 12345", fill="black")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def main() -> int:
    show_config()
    if not settings.llm_configured:
        print("\n未找到 Key：cp .env.example .env，写入 SILICONFLOW_API_KEY=sk-... 后重试。")
        return 1

    headers = {
        "Authorization": f"Bearer {settings.llm_api_key.strip()}",
        "Content-Type": "application/json",
    }

    ids = fetch_models(headers)
    if ids is None:
        return 2
    if "--models" in sys.argv:
        print_catalog(ids)
        return 0
    if "--candidates" in sys.argv:
        return 0 if check_candidates(headers) else 2

    models_ok, _ = check_models(ids)
    print("\n== 连通性 ==")
    chat_ok = check_chat(headers)
    embed_ok = check_embedding(headers)
    rerank_ok = check_rerank(headers)
    ocr_ok = check_ocr() if "--ocr" in sys.argv else True
    if "--ocr" not in sys.argv and settings.ocr_enabled:
        print("  SKIP vision    加 --ocr 可实测一次（会产生一次调用）")

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
