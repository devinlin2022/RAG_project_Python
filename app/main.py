"""Entry point: one process serving both the Gradio UI and the FastAPI routes.

Works the same way locally, in Docker and in Colab. Only two things differ and
both are detected, not configured: the bind address and whether a tunnel is
opened (a notebook has no reachable localhost).
"""

import importlib
import logging
import os
import sys
from pathlib import Path

# `python app/main.py` puts app/ on sys.path, so `import app.api` fails there.
# Colab users type exactly that, so make both spellings work.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import uvicorn
from fastapi import FastAPI

from app.api import public, router
from app.auth import setup_logging
from app.state import get_service
from app.ui import build_ui
from rag.config import VERSION, settings

setup_logging()
log = logging.getLogger("rag-agent")

# Parsers are optional at import time so a missing wheel degrades to "this
# format is unavailable" instead of "the app will not start".
OPTIONAL_DEPS = {
    "pypdfium2": "PDF 渲染与文本抽取",
    "pdfplumber": "表格识别",
    "docx": "Word 解析（python-docx）",
    "pptx": "PPT 解析（python-pptx）",
    "openpyxl": "Excel 解析",
    "jieba": "中文分词（缺失则退化为二元组）",
}


def _in_colab() -> bool:
    return "google.colab" in sys.modules or bool(os.environ.get("COLAB_RELEASE_TAG"))


def bind_host() -> str:
    """0.0.0.0 when a tunnel or Colab needs it, else the configured address."""
    if settings.ngrok_enabled or settings.ngrok_authtoken.strip() or _in_colab():
        return "0.0.0.0"
    return settings.app_host


def check_dependencies() -> list[str]:
    missing = []
    for module, purpose in OPTIONAL_DEPS.items():
        try:
            importlib.import_module(module)
        except ImportError:
            missing.append(f"{module}（{purpose}）")
    return missing


def create_app() -> FastAPI:
    service = get_service()
    api = FastAPI(
        title="rag-agent",
        version=VERSION,
        description="文档助手：多格式解析、混合检索、精排与可核验引用",
    )
    api.include_router(public)
    api.include_router(router)
    return gr_mount(api, service)


def gr_mount(api: FastAPI, service):
    import gradio as gr

    return gr.mount_gradio_app(api, build_ui(service), path="/", auth=settings.ui_auth)


app = create_app()


def open_tunnel(port: int) -> str | None:
    """Open an ngrok tunnel. Never fatal: a local run still works."""
    token = (settings.ngrok_authtoken or "").strip()
    if not settings.ngrok_enabled and not token:
        return None
    if not token:
        print("  ngrok    : 已启用但未配置 NGROK_AUTHTOKEN，跳过公网隧道")
        return None
    try:
        from pyngrok import ngrok
    except ImportError:
        print("  ngrok    : 未安装 pyngrok，跳过公网隧道（pip install pyngrok）")
        return None

    try:
        ngrok.set_auth_token(token)
        # The token above is enough; a saved config from a previous run can
        # otherwise point at a dead tunnel and fail with a confusing error.
        for tunnel in ngrok.get_tunnels():
            ngrok.disconnect(tunnel.public_url)
        tunnel = ngrok.connect(port, "http")
        return tunnel.public_url
    except Exception as exc:  # quota, network, bad token
        log.warning("ngrok tunnel failed: %s", exc)
        print(f"  ngrok    : 隧道建立失败，仅本地可用（{exc}）")
        print("             常见原因：token 无效、免费额度用尽、或本机 7890 端口被占用")
        return None


def main() -> None:
    service = get_service()
    stats = service.library.stats()
    host = bind_host()
    port = settings.app_port

    print(f"rag-agent v{VERSION}")
    print(f"  base_url : {settings.llm_base_url}")
    print(f"  chat     : {settings.llm_model}")
    print(f"  embed    : {settings.embed_model} {'可用' if settings.embed_ready else '不可用'}")
    print(f"  rerank   : {settings.rerank_model} {'可用' if settings.rerank_ready else '关闭'}")
    print(f"  vision   : {settings.vision_model} {'启用' if settings.ocr_enabled else '关闭'}")
    print(f"  api_key  : {'已配置' if settings.llm_configured else '未配置（问答会提示）'}")
    print(f"  index    : {stats['docs']} 篇 / {stats['chunks']} 块 / {stats['vectors']} 向量")
    print(f"  auth     : API Key {'必需' if settings.app_api_key else '未启用'}"
          f"　｜　界面登录 {'启用' if settings.ui_auth else '未启用'}")
    print(f"  页面     : http://{host if host != '0.0.0.0' else '127.0.0.1'}:{port}")
    print(f"  接口文档 : http://{host if host != '0.0.0.0' else '127.0.0.1'}:{port}/docs")

    missing = check_dependencies()
    if missing:
        print("  缺少依赖 : " + "、".join(missing))
        print("             pip install -r requirements.txt")

    if not settings.llm_configured:
        print("\n  ⚠ 未找到 API Key：在 .env 写入 SILICONFLOW_API_KEY=sk-... 后重启")
        print("    （没有 Key 也能解析、切分和关键词检索，只是不能回答）\n")

    public_url = open_tunnel(port)
    if public_url:
        print("\n" + "=" * 64)
        print(f"  公网访问: {public_url}")
        print("=" * 64 + "\n")

    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
