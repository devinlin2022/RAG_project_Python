"""Every outbound client against the fake platform."""

import socket
import threading
import time

import pytest
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from rag.config import settings
from rag.embed import client as embed_client
from rag.generate import llm
from rag.retrieve import reranker


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class Platform:

    def __init__(self):
        self.calls: dict[str, list[float]] = {}
        self.scripts: dict[str, list[int]] = {}
        self.lock = threading.Lock()
        self.app = FastAPI()
        self._routes()

    def _next(self, path: str) -> int:
        with self.lock:
            self.calls.setdefault(path, []).append(time.monotonic())
            script = self.scripts.get(path) or []
            return script.pop(0) if script else 200

    def hits(self, path: str) -> int:
        with self.lock:
            return len(self.calls.get(path, []))

    def _routes(self):
        @self.app.post("/v1/chat/completions")
        async def chat(request: Request):
            body = await request.json()
            status = self._next("chat")
            if status != 200:
                return JSONResponse(
                    {"code": 30014, "message": "Token is invalid."}, status_code=status
                )
            if not body.get("stream"):
                return {
                    "choices": [{"message": {"content": "答案是 30 天。"}}],
                    "usage": {"prompt_tokens": 12, "completion_tokens": 4},
                }

            def gen():
                yield 'data: {"choices":[{"delta":{"content":"答案"}}]}\n\n'
                yield 'data: {"choices":[{"delta":{"content":"是 30 天。"}}]}\n\n'
                yield 'data: {"usage":{"prompt_tokens":12,"completion_tokens":4}}\n\n'
                yield "data: [DONE]\n\n"

            return StreamingResponse(gen(), media_type="text/event-stream")

        @self.app.post("/v1/embeddings")
        async def embeddings(request: Request):
            body = await request.json()
            status = self._next("embed")
            if status != 200:
                return JSONResponse({"message": "Token is invalid."}, status_code=status)
            items = body["input"] if isinstance(body["input"], list) else [body["input"]]
            return {
                "data": [{"index": index, "embedding": [0.1 * (index + 1), 0.2, 0.3]}
                         for index in range(len(items))],
                "usage": {"prompt_tokens": 5 * len(items)},
            }

        @self.app.post("/v1/rerank")
        async def rerank(request: Request):
            body = await request.json()
            status = self._next("rerank")
            if status != 200:
                return JSONResponse({"message": "rate limited"}, status_code=status)
            docs = body["documents"]
            return {
                "results": [
                    {"index": index, "relevance_score": 1.0 - 0.1 * index}
                    for index in range(len(docs))
                ]
            }


@pytest.fixture()
def platform(monkeypatch):
    fake = Platform()
    port = free_port()
    server = uvicorn.Server(
        uvicorn.Config(fake.app, host="127.0.0.1", port=port, log_level="error")
    )
    threading.Thread(target=server.run, daemon=True).start()
    for _ in range(100):
        if _ready(port):
            break
        time.sleep(0.05)

    base = f"http://127.0.0.1:{port}/v1"
    monkeypatch.setattr(settings, "llm_base_url", base)
    monkeypatch.setattr(settings, "embed_base_url", "")
    monkeypatch.setattr(settings, "llm_api_key", "sk-test-key-abcd")
    monkeypatch.setattr(settings, "http_min_interval", 0.0)
    monkeypatch.setattr(settings, "http_retries", 2)
    embed_client.reset_breaker()
    yield fake
    server.should_exit = True
    time.sleep(0.1)


def _ready(port: int) -> bool:
    with socket.socket() as sock:
        sock.settimeout(0.2)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def test_401_fails_once_with_actionable_message(platform):
    platform.scripts["chat"] = [401]
    with pytest.raises(llm.LLMError) as caught:
        llm.chat([{"role": "user", "content": "hi"}])
    message = str(caught.value)
    assert "401" in message and "尾号 abcd" in message
    assert "共用" in message
    assert platform.hits("chat") == 1, "401 不该重试，用户等 3 轮还是同一个错"


def test_429_then_success_returns_answer(platform):
    platform.scripts["chat"] = [429]
    answer, usage = llm.chat([{"role": "user", "content": "hi"}])
    assert "30 天" in answer
    assert usage["prompt_tokens"] == 12
    assert platform.hits("chat") == 2


def test_stream_retries_before_first_byte(platform):
    platform.scripts["chat"] = [503]
    pieces = list(llm.chat_stream([{"role": "user", "content": "hi"}]))
    assert "".join(pieces) == "答案是 30 天。"
    assert platform.hits("chat") == 2


def test_stream_401_raises_before_yielding(platform):
    platform.scripts["chat"] = [401]
    with pytest.raises(llm.LLMError):
        list(llm.chat_stream([{"role": "user", "content": "hi"}]))
    assert platform.hits("chat") == 1


def test_rerank_scores_are_sorted(platform):
    pairs = reranker.rerank("账期", ["a", "b", "c"])
    assert [index for index, _ in pairs] == [0, 1, 2]
    assert pairs[0][1] > pairs[-1][1]


def test_rerank_429_raises_rate_limit_message(platform):
    platform.scripts["rerank"] = [429, 429, 429]
    with pytest.raises(reranker.RerankError) as caught:
        reranker.rerank("账期", ["a", "b"])
    assert "限流" in str(caught.value)
    assert platform.hits("rerank") == 3, "retries=2 → 最多 3 次"


def test_dead_reranker_keeps_fusion_order():
    from rag.index.library import Record
    from rag.retrieve.retriever import Hit, rerank_hits

    def broken(*_args, **_kwargs):
        raise RuntimeError("精排不可用：Key 被拒绝（401）")

    hits = [
        Hit(
            record=Record(
                doc_id="d1", filename="a.pdf", idx=index, text=f"第{index}块",
                page_from=index + 1, page_to=index + 1,
            ),
            score=1.0 - index * 0.1,
        )
        for index in range(3)
    ]
    kept, info = rerank_hits("q", hits, top_k=2, reranker=broken)
    assert info["used"] is False and "401" in info["error"]
    assert len(kept) == 2
    assert [hit.record.idx for hit in kept] == [0, 1], "应按融合分保持原顺序"


def test_embed_401_trips_breaker_then_reset_restores(platform):
    platform.scripts["embed"] = [401]
    with pytest.raises(embed_client.EmbedError):
        embed_client.embed_texts(["测试"])
    paused, reason, left = embed_client.breaker_state()
    assert paused and left > 0 and "401" in reason
    assert platform.hits("embed") == 1

    with pytest.raises(embed_client.EmbedError) as caught:
        embed_client.embed_texts(["再来一次"])
    assert "暂停" in str(caught.value) and platform.hits("embed") == 1

    embed_client.reset_breaker()
    assert embed_client.breaker_state()[0] is False
    vectors = embed_client.embed_texts(["恢复了"])
    assert len(vectors) == 1 and len(vectors[0]) == 3


def test_embed_batches_keep_order(platform):
    texts = [f"第{i}段" for i in range(5)]
    vectors = embed_client.embed_texts(texts, batch=2)
    assert len(vectors) == 5
    assert [round(vector[0], 2) for vector in vectors] == [0.1, 0.2, 0.1, 0.2, 0.1]
