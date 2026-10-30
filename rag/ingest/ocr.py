"""Vision calls for scanned pages (OCR) and diagrams (caption)."""

import logging
import time
from dataclasses import dataclass

import requests

from rag.config import settings
from rag.ingest.render import image_block
from rag.net import post_json

log = logging.getLogger(__name__)

OCR_PROMPT = (
    "你是文档 OCR 引擎。提取图中所有可见文字，要求：\n"
    "1. 保持阅读顺序，按行输出，不要合并不同区块；\n"
    "2. 表格必须输出为 Markdown 表格，保留表头与单元格对应关系；\n"
    "3. 保留条款编号（第X条、一、（一）、1.1 等）原样；\n"
    "4. 只输出识别结果，不要任何解释、前言或总结；\n"
    "5. 无法辨认的字符用 □ 代替，不要猜测。"
)

# OCR only yields labels; a diagram needs the arrows, so ask for edges explicitly.
FIGURE_PROMPT = (
    "你是文档图表分析助手。请描述这张图，要求：\n"
    "1. 先说图表类型（流程图 / 组织架构 / 时序图 / 截图 / 照片 / 图表）；\n"
    "2. 列出图中所有节点、角色或数据项；\n"
    "3. 用 A -> B 的形式逐条写出箭头关系与分支条件，这一步最重要；\n"
    "4. 如果图中有文字结论、图例或标注，逐条抄录；\n"
    "5. 控制在 300 字以内，不要推测图中不存在的信息。"
)


class OcrError(Exception):
    """OCR or caption call failed."""


@dataclass
class OcrResult:
    text: str
    engine: str
    ms: int
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def empty(self) -> bool:
        return not self.text.strip()


def vision_chat(png: bytes, prompt: str, model: str) -> OcrResult:
    api_key = settings.ocr_api_key_resolved
    if not api_key:
        raise OcrError("未配置 SILICONFLOW_API_KEY，无法调用视觉模型（OCR 与问答共用同一把 Key）")

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}, image_block(png)],
            }
        ],
        "temperature": 0.0,
        "max_tokens": 2048,
        "stream": False,
    }
    started = time.perf_counter()
    try:
        response = post_json(
            settings.ocr_base_url_resolved + "/chat/completions",
            {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            payload,
            settings.ocr_timeout,
            min_interval=settings.ocr_min_interval,
            retries=settings.http_retries,
        )
    except requests.Timeout as exc:
        raise OcrError(f"视觉模型超时（{settings.ocr_timeout}s），可调大 OCR_TIMEOUT") from exc
    except Exception as exc:
        raise OcrError(f"视觉模型请求失败：{exc}") from exc

    elapsed = int((time.perf_counter() - started) * 1000)

    if response.status_code == 401:
        raise OcrError(
            f"视觉模型拒绝了这个 Key（401）：当前模型 {model}、地址 {settings.ocr_base_url_resolved}。"
            "请确认该 Key 已开通视觉模型权限"
        )
    if response.status_code == 402:
        raise OcrError("账户余额不足（402），请先在控制台充值")
    if response.status_code == 429:
        raise OcrError("触发限流（429），请调大 OCR_MIN_INTERVAL 或减少 OCR_MAX_PAGES")
    if response.status_code == 404:
        raise OcrError(f"视觉模型不存在（404）：{model}，请核对 OCR_MODEL 或 VISION_MODEL")
    if response.status_code >= 400:
        raise OcrError(f"视觉模型返回 {response.status_code}：{response.text[:200]}")

    try:
        body = response.json()
        text = body["choices"][0]["message"]["content"] or ""
        usage = body.get("usage") or {}
    except Exception as exc:
        raise OcrError(f"视觉模型响应无法解析：{response.text[:200]}") from exc

    return OcrResult(
        text=text.strip(),
        engine=model,
        ms=elapsed,
        prompt_tokens=int(usage.get("prompt_tokens") or 0),
        completion_tokens=int(usage.get("completion_tokens") or 0),
    )


def ocr_page(png: bytes) -> OcrResult:
    return vision_chat(png, OCR_PROMPT, settings.ocr_model)


def describe_figure(png: bytes) -> OcrResult:
    return vision_chat(png, FIGURE_PROMPT, settings.vision_model)


def ocr_local(png: bytes) -> OcrResult:
    """Offline fallback. Optional: pip install rapidocr-onnxruntime."""
    try:
        from rapidocr_onnxruntime import RapidOCR
    except ImportError as exc:
        raise OcrError(
            "未安装本地 OCR：pip install rapidocr-onnxruntime，"
            "或在 .env 中配置 OCR_MODEL 使用云端识别"
        ) from exc

    import io

    from PIL import Image
    import numpy as np

    started = time.perf_counter()
    image = np.array(Image.open(io.BytesIO(png)).convert("RGB"))
    result, _ = RapidOCR()(image)
    elapsed = int((time.perf_counter() - started) * 1000)
    return OcrResult(
        text="\n".join(item[1] for item in (result or [])).strip(),
        engine="rapidocr",
        ms=elapsed,
    )
