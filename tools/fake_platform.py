"""A stand-in for the SiliconFlow API, for offline verification.

The real platform is the one thing we cannot make deterministic: it needs a
valid key, it rate-limits, and it occasionally answers 500. This fake answers the
four endpoints this project uses (`/chat/completions` with and without stream,
`/embeddings`, `/rerank`) and can be told to fail on purpose, so "401 is not
retried", "429 backs off", "rerank dying does not kill the answer" become
assertions instead of anecdotes.

Used by `python -m tools.smoke_offline` and by `tests/test_clients.py`.
"""

from __future__ import annotations

import socket
import threading
import time

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def port_ready(port: int, host: str = "127.0.0.1", timeout: float = 0.3) -> bool:
    with socket.socket() as sock:
        sock.settimeout(timeout)
        return sock.connect_ex((host, port)) == 0


class FakePlatform:
    """每个路径一份脚本，按顺序吐出状态码；到达时刻全部记账。"""

    def __init__(self, dim: int = 8):
        self.dim = dim
        self.calls: dict[str, list[float]] = {}
        self.scripts: dict[str, list[int]] = {}
        self.lock = threading.Lock()
        self.port = 0
        self._server = None
        self.app = self._build()

    # ── 记账 ────────────────────────────────────────────────

    def _next(self, path: str) -> int:
        with self.lock:
            self.calls.setdefault(path, []).append(time.monotonic())
            script = self.scripts.get(path) or []
            return script.pop(0) if script else 200

    def script(self, path: str, codes: list[int]) -> None:
        self.scripts[path] = list(codes)

    def hits(self, path: str) -> int:
        with self.lock:
            return len(self.calls.get(path, []))

    def gaps(self, path: str) -> list[float]:
        with self.lock:
            stamps = sorted(self.calls.get(path, []))
        return [round(b - a, 3) for a, b in zip(stamps, stamps[1:])]

    def reset(self) -> None:
        with self.lock:
            self.calls.clear()
            self.scripts.clear()

    # ── 路由 ────────────────────────────────────────────────

    def _build(self) -> FastAPI:
        app = FastAPI()

        @app.post("/v1/chat/completions")
        async def chat(request: Request):
            body = await request.json()
            status = self._next("chat")
            if status != 200:
                return JSONResponse(
                    {"code": 30014, "message": "Token is invalid."}, status_code=status
                )

            messages = body.get("messages") or []
            prompt = "".join(
                part.get("text", "")
                for message in messages
                for part in (
                    message.get("content")
                    if isinstance(message.get("content"), list)
                    else [{"text": str(message.get("content") or "")}]
                )
            )

            # 出题请求返回 JSON 数组，问答请求返回一句带引用的答案：
            # 两种调用共用一个端点，靠提示词区分。
            if "只输出 JSON 数组" in prompt:
                content = (
                    '[{"q": "付款账期最长是多少天？", "answer": "30 天"},'
                    ' {"q": "紧急采购后多久要补办手续？", "answer": "5 个工作日内"}]'
                )
            else:
                content = "付款账期最长为 30 天 [1]。紧急采购需在 5 个工作日内补办审批 [2]。"

            if not body.get("stream"):
                return {
                    "choices": [{"message": {"content": content}}],
                    "usage": {"prompt_tokens": 120, "completion_tokens": 30},
                }

            def gen():
                for piece in (content[: len(content) // 2], content[len(content) // 2 :]):
                    yield (
                        'data: {"choices":[{"delta":{"content":'
                        + _json_str(piece)
                        + "}}]}\n\n"
                    )
                yield 'data: {"usage":{"prompt_tokens":120,"completion_tokens":30}}\n\n'
                yield "data: [DONE]\n\n"

            return StreamingResponse(gen(), media_type="text/event-stream")

        @app.post("/v1/embeddings")
        async def embeddings(request: Request):
            body = await request.json()
            status = self._next("embed")
            if status != 200:
                return JSONResponse({"message": "Token is invalid."}, status_code=status)
            items = body["input"] if isinstance(body["input"], list) else [body["input"]]
            return {
                "data": [
                    {"index": index, "embedding": _vector(text, self.dim)}
                    for index, text in enumerate(items)
                ],
                "usage": {"prompt_tokens": 7 * len(items)},
            }

        @app.post("/v1/rerank")
        async def rerank(request: Request):
            body = await request.json()
            status = self._next("rerank")
            if status != 200:
                return JSONResponse({"message": "rate limited"}, status_code=status)
            docs = body.get("documents") or []
            return {
                "results": [
                    {"index": index, "relevance_score": 1.0 - 0.05 * index}
                    for index in range(len(docs))
                ]
            }

        return app

    # ── 生命周期 ────────────────────────────────────────────

    def start(self) -> str:
        self.port = free_port()
        self._server = uvicorn.Server(
            uvicorn.Config(self.app, host="127.0.0.1", port=self.port, log_level="error")
        )
        threading.Thread(target=self._server.run, daemon=True).start()
        for _ in range(100):
            if port_ready(self.port):
                break
            time.sleep(0.05)
        return f"http://127.0.0.1:{self.port}/v1"

    def stop(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
            self._server = None


def _json_str(text: str) -> str:
    import json

    return json.dumps(text, ensure_ascii=False)


def _vector(text: str, dim: int) -> list[float]:
    """确定性向量：同样的文本给同样的向量，余弦才有意义。"""
    import hashlib

    seed = hashlib.sha256(text.encode("utf-8")).digest()
    raw = [(seed[index % len(seed)] - 127.5) / 127.5 for index in range(dim)]
    norm = sum(value * value for value in raw) ** 0.5 or 1.0
    return [round(value / norm, 6) for value in raw]
