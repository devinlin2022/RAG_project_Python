"""Entry point: one process serving both the Gradio UI and the FastAPI routes."""

import logging

import gradio as gr
import uvicorn
from fastapi import FastAPI

from app.api import router
from app.state import STORE
from app.ui import build_ui
from rag.config import VERSION, settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
log = logging.getLogger("rag-agent")

HOST = "127.0.0.1"
PORT = 7860


def create_app() -> FastAPI:
    api = FastAPI(
        title="rag-agent",
        version=VERSION,
        description="按页抽取与页级问答（V1，不含检索）",
    )
    api.include_router(router)
    return gr.mount_gradio_app(api, build_ui(STORE), path="/")


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
    print(f"rag-agent v{VERSION}")
    print(f"  base_url : {settings.llm_base_url}")
    print(f"  model    : {settings.llm_model}")
    print(f"  api_key  : {'已配置' if settings.llm_configured else '未配置（问答会提示）'}")
    print(f"  页面     : http://{HOST}:{PORT}")
    print(f"  接口文档 : http://{HOST}:{PORT}/docs")

    public_url = open_tunnel()
    if public_url:
        print(f"\n公网访问链接 (Share URL): {public_url}\n")

    uvicorn.run(app, host=HOST, port=PORT, log_level="info")


if __name__ == "__main__":
    main()
