"""Colab 一键启动：写 .env、起服务、打印可点开的地址。

由 colab.ipynb 调用，但**不依赖 Colab**——笔记本跑在本机也一样能用，
只有开隧道那一步是 notebook 才需要的。

设计上只做三件事，且都能重复执行：
1. 有 .env 就改，没有就从 .env.example 复制一份（注释一起带过去）
2. 起 uvicorn 线程（主线程留出来做交互，不再 `!python run.py` 卡死一格）
3. 端口已经在听就不再起第二个，直接把地址重新打一遍
"""

from __future__ import annotations

import getpass
import os
import socket
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = ROOT / ".env"
EXAMPLE_PATH = ROOT / ".env.example"

KEY_NAME = "SILICONFLOW_API_KEY"
TOKEN_NAME = "NGROK_AUTHTOKEN"

_SERVER = None
_TUNNEL = ""


# ── .env 读写 ────────────────────────────────────────────────


def parse_env(text: str) -> dict[str, str]:
    """只认 KEY=VALUE，注释和空行忽略；引号去掉。"""
    out: dict[str, str] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        out[key.strip()] = value.strip().strip('"').strip("'")
    return out


def patch_env(text: str, updates: dict[str, str]) -> str:
    """就地替换 KEY=VALUE，保留原有注释、顺序和空行。

    不直接 `>>` 追加：dotenv 是"后写的赢"，同一个键出现两行时用户读 .env
    会以为上面那行还有效，排查起来非常费劲。
    """
    if not updates:
        return text if text.endswith("\n") or not text else text + "\n"

    seen: set[str] = set()
    lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            key = stripped.partition("=")[0].strip()
            if key in updates:
                lines.append(f"{key}={updates[key]}")
                seen.add(key)
                continue
        lines.append(line)

    missing = [k for k in updates if k not in seen]
    if missing:
        # 空文件不写分隔注释，否则从零生成的 .env 顶上会挂一行没上下文的标题
        if lines:
            if lines[-1].strip():
                lines.append("")
            lines.append("# ── 由 colab.ipynb 写入 ──")
        lines.extend(f"{k}={updates[k]}" for k in missing)

    return "\n".join(lines).rstrip("\n") + "\n"


def write_env(api_key: str = "", ngrok_token: str = "", force: bool = False) -> dict:
    """写 .env。默认只补空值，不覆盖用户手填的内容。"""
    if not EXAMPLE_PATH.exists():
        raise FileNotFoundError(f"缺少模板文件：{EXAMPLE_PATH}")

    existed = ENV_PATH.exists()
    text = ENV_PATH.read_text(encoding="utf-8") if existed else EXAMPLE_PATH.read_text(encoding="utf-8")
    current = parse_env(text)

    updates: dict[str, str] = {}
    if api_key and (force or not current.get(KEY_NAME)):
        updates[KEY_NAME] = api_key
    if ngrok_token and (force or not current.get(TOKEN_NAME)):
        updates[TOKEN_NAME] = ngrok_token

    if updates:
        ENV_PATH.write_text(patch_env(text, updates), encoding="utf-8")
        try:
            os.chmod(ENV_PATH, 0o600)  # 里面是 Key，别让同机其他用户读
        except OSError:
            pass

    merged = {**current, **updates}
    return {
        "created": ENV_PATH.exists() and not existed,
        "changed": sorted(updates),
        "has_key": bool(merged.get(KEY_NAME)),
        "has_token": bool(merged.get(TOKEN_NAME)),
        "values": merged,
    }


# ── 取密钥：环境变量 → Colab Secrets → 手输 ───────────────────


def read_secret(name: str) -> str:
    value = (os.environ.get(name) or "").strip()
    if value:
        return value
    try:
        from google.colab import userdata  # type: ignore

        return (userdata.get(name) or "").strip()
    except Exception:
        # 没有这个 Secret、或用户还没点"允许访问"，都走手输
        return ""


def _ask(prompt: str) -> str:
    try:
        return getpass.getpass(prompt).strip()
    except Exception:
        return ""


# ── 端口与进程 ──────────────────────────────────────────────


def port_open(port: int, host: str = "127.0.0.1", timeout: float = 0.4) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        return sock.connect_ex((host, int(port))) == 0


def _wait_port(port: int, seconds: float = 90.0) -> bool:
    deadline = time.time() + seconds
    while time.time() < deadline:
        if port_open(port):
            return True
        time.sleep(0.4)
    return False


def _first_tunnel() -> str:
    try:
        from pyngrok import ngrok

        tunnels = ngrok.get_tunnels()
        return tunnels[0].public_url if tunnels else ""
    except Exception:
        return ""


def stop() -> None:
    """停掉本 kernel 起的服务，下次 boot 才能换端口/换配置。"""
    global _SERVER, _TUNNEL
    if _SERVER is not None:
        _SERVER.should_exit = True
        _SERVER = None
    _TUNNEL = ""


# ── 启动 ────────────────────────────────────────────────────


def serve(port: int | None = None, tunnel: bool = True) -> str:
    global _SERVER, _TUNNEL

    from rag.config import settings

    port = int(port or settings.app_port)

    if _SERVER is not None:
        return _TUNNEL or f"http://127.0.0.1:{port}"
    if port_open(port):
        # 上一格已经在跑（或上个 kernel 留下的）——不要再绑一次
        _TUNNEL = _first_tunnel()
        return _TUNNEL or f"http://127.0.0.1:{port}"

    try:
        import nest_asyncio

        nest_asyncio.apply()
    except ImportError:
        pass

    import uvicorn

    from app.main import app, bind_host, open_tunnel

    server = uvicorn.Server(
        uvicorn.Config(app, host=bind_host(), port=port, log_level="info", access_log=False)
    )
    threading.Thread(target=server.run, name="uvicorn", daemon=True).start()
    _SERVER = server

    if not _wait_port(port):
        _SERVER = None
        raise RuntimeError(f"启动超时：{port} 端口 90 秒内没起来，看上面日志里的报错")

    url = f"http://127.0.0.1:{port}"
    if tunnel:
        _TUNNEL = open_tunnel(port) or ""
        if _TUNNEL:
            url = _TUNNEL
    return url


def _sync_live_settings(env_values: dict[str, str]) -> None:
    """.env 改了但模块已经 import 过：把新值同步进内存里的 settings。

    界面的"粘贴 Key"本来就是这么改的（app/ui.py），这些字段都是运行时读，
    所以不用重启 kernel——重启一次 Colab 会话要重装依赖，代价太大。
    """
    if "rag.config" not in sys.modules:
        return
    from rag.config import settings

    key = env_values.get(KEY_NAME, "")
    if key and key != settings.llm_api_key:
        settings.llm_api_key = key
        print(f"  已同步 Key（尾号 {key[-4:]}），无需重启会话")
    token = env_values.get(TOKEN_NAME, "")
    if token and token != settings.ngrok_authtoken:
        settings.ngrok_authtoken = token


def _banner(url: str) -> str:
    from rag.config import VERSION, settings

    stats = _stats()
    ocr_on = "就绪" if settings.ocr_ready else settings.vision_state[1]
    lines = [
        "",
        "=" * 66,
        f"  文档助手 v{VERSION}",
        f"  地址    {url}",
        f"  模型    问答 {settings.llm_model.split('/')[-1]}"
        f" ｜ 向量 {'就绪' if settings.embed_ready else '关闭'}"
        f" ｜ 精排 {'就绪' if settings.rerank_ready else '关闭'}"
        f" ｜ 视觉 {ocr_on}",
        f"  索引    {stats}",
        "=" * 66,
    ]
    if url.startswith("http://127.0.0.1"):
        lines.insert(-1, "  注意    本地地址在 Colab 里点不开，需要隧道或端口代理")
    if not settings.llm_configured:
        lines.insert(-1, "  注意    没有 Key：能解析、能关键词检索，但不能问答")
    return "\n".join(lines)


def _stats() -> str:
    try:
        from app.state import get_service

        info = get_service().library.stats()
        return f"{info['docs']} 篇 / {info['chunks']} 块 / {info['vectors']} 向量"
    except Exception:
        return "未统计"


def boot(
    api_key: str = "",
    ngrok_token: str = "",
    port: int | None = None,
    ask: bool = True,
) -> str:
    """笔记本里唯一的入口：拿 Key → 写 .env → 起服务 → 打印地址。"""
    if not api_key:
        api_key = read_secret(KEY_NAME)
    if not api_key and ask:
        api_key = _ask("硅基流动 API Key（sk- 开头，直接回车跳过）：")

    if not ngrok_token:
        ngrok_token = read_secret(TOKEN_NAME)
    if not ngrok_token and ask:
        ngrok_token = _ask("ngrok token（可留空，留空则只能用Colab端口代理）：")

    info = write_env(api_key=api_key, ngrok_token=ngrok_token)
    if info["created"]:
        print("  已从 .env.example 生成 .env")
    if info["changed"]:
        print("  已写入：" + "、".join(info["changed"]))

    _sync_live_settings(info["values"])

    url = serve(port=port)
    print(_banner(url))
    return url
