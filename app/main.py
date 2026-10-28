"""Entry point: one process serving both the Gradio UI and the FastAPI routes."""

import logging
import time

import gradio as gr
import uvicorn
from fastapi import FastAPI, Request

from app.api import public, router
from app.auth import setup_logging
from app.state import get_service
from app.ui import build_ui
from rag.config import VERSION, settings

setup_logging()
log = logging.getLogger("rag-agent")

HOST = settings.app_host
PORT = settings.app_port


def create_app() -> FastAPI:
    service = get_service()
    api = FastAPI(
        title="rag-agent",
        version=VERSION,
        description="PDF 助手：按页抽取、混合检索、精排与带引用问答",
    )
    api.include_router(public)
    api.include_router(router)

    @api.middleware("http")
    async def log_requests(request: Request, call_next):
        started = time.perf_counter()
        response = await call_next(request)
        if request.url.path != "/health":
            log.info(
                "%s %s -> %s in %.0fms",
                request.method,
                request.url.path,
                response.status_code,
                (time.perf_counter() - started) * 1000,
            )
        return response

    return gr.mount_gradio_app(
        api, build_ui(service), path="/", auth=settings.ui_auth
    )


app = create_app()


def open_tunnel() -> str | None:
    """Open an ngrok tunnel when configured. Never fatal: local use still works."""
    if not settings.ngrok_enabled:
        return None
    token = settings.ngrok_authtoken.strip()
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
        tunnel = ngrok.connect(PORT)
        return tunnel.public_url
    except Exception as exc:  # quota, network, bad token
        log.warning("ngrok tunnel failed: %s", exc)
        print(f"  ngrok    : 隧道建立失败，仅本地可用（{exc}）")
        return None


def main() -> None:
    service = get_service()
    stats = service.library.stats()
    print(f"rag-agent v{VERSION}")
    print(f"  base_url : {settings.llm_base_url}")
    print(f"  model    : {settings.llm_model}")
    print(f"  api_key  : {'已配置' if settings.llm_configured else '未配置（问答会提示）'}")
    print(f"  embed    : {settings.embed_model} {'可用' if settings.embed_ready else '不可用'}")
    print(f"  rerank   : {settings.rerank_model} {'可用' if settings.rerank_ready else '关闭'}")
    print(f"  index    : {stats['docs']} 篇 / {stats['chunks']} 块 / {stats['vectors']} 向量")
    print(f"  auth     : API Key {'必需' if settings.app_api_key else '未启用'}"
          f"　｜　界面登录 {'启用' if settings.ui_auth else '未启用'}")
    print(f"  页面     : http://{HOST}:{PORT}")
    print(f"  接口文档 : http://{HOST}:{PORT}/docs")

    public_url = open_tunnel()
    if public_url:
        print(f"\n公网访问链接 (Share URL): {public_url}\n")

    uvicorn.run(app, host=HOST, port=PORT, log_level="info")


if __name__ == "__main__":
    main()
