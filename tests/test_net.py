"""网络层的行为测试：节流真的错开了、401 不重试、429/5xx 才会重试。

用一台本地假服务器记账（每条请求的到达时间 + 次数），不碰真实平台：
这样"间隔够不够"和"重试了几次"都是可断言的事实，而不是看日志猜。
"""

import socket
import threading
import time

import pytest
import requests
import uvicorn
from fastapi import FastAPI, Response

from rag import net


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class FakePlatform:
    """记录到达时刻的假硅基流动：按 script 依次返回状态码。"""

    def __init__(self):
        self.hits: list[float] = []
        self.script: list[int] = []
        self.retry_after: str | None = None
        self.lock = threading.Lock()
        self.app = FastAPI()

        @self.app.post("/call")
        def call():
            with self.lock:
                self.hits.append(time.monotonic())
                status = self.script.pop(0) if self.script else 200
                header = self.retry_after
            if status == 200:
                return {"ok": True}
            return Response(
                content='{"code": 30014, "message": "Token is invalid."}',
                status_code=status,
                media_type="application/json",
                headers={"Retry-After": header} if header else None,
            )

        @self.app.post("/slow")
        def slow():
            with self.lock:
                self.hits.append(time.monotonic())
            time.sleep(0.6)
            return {"ok": True}

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}/"

    def count(self) -> int:
        with self.lock:
            return len(self.hits)


@pytest.fixture()
def platform():
    fake = FakePlatform()
    fake.port = free_port()
    server = uvicorn.Server(
        uvicorn.Config(fake.app, host="127.0.0.1", port=fake.port, log_level="error")
    )
    threading.Thread(target=server.run, daemon=True).start()
    for _ in range(100):
        if net.port_ready(fake.port):
            break
        time.sleep(0.05)
    net.reset_pacing()
    yield fake
    server.should_exit = True
    time.sleep(0.1)


def post(fake, path="call", **kwargs):
    kwargs.setdefault("timeout", 5)
    return net.post_json(
        fake.url + path,
        {"Authorization": "Bearer sk-test"},
        {"input": "x"},
        min_interval=kwargs.pop("min_interval", 0.0),
        retries=kwargs.pop("retries", 3),
        base_delay=kwargs.pop("base_delay", 0.01),
        **kwargs,
    )


def test_concurrent_calls_are_spaced(platform):
    """4 个线程同时打，到达时间必须被拉开到 min_interval 以上。"""
    net.reset_pacing()
    threads = [
        threading.Thread(target=lambda: post(platform, min_interval=0.2)) for _ in range(4)
    ]
    started = time.monotonic()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    elapsed = time.monotonic() - started

    stamps = sorted(platform.hits)
    gaps = [b - a for a, b in zip(stamps, stamps[1:])]
    assert len(stamps) == 4
    assert min(gaps) >= 0.15, f"出现了并发突刺：{gaps}"
    assert elapsed >= 0.55, "4 次调用 × 0.2s 至少该花 0.6s"


def test_401_is_not_retried(platform):
    platform.script = [401]
    response = post(platform)
    assert response.status_code == 401
    assert platform.count() == 1, "401 是永久错误，重试只会让用户多等几秒"


def test_402_is_not_retried(platform):
    platform.script = [402]
    assert post(platform).status_code == 402
    assert platform.count() == 1


def test_429_is_retried_then_succeeds(platform):
    platform.script = [429]
    response = post(platform)
    assert response.status_code == 200
    assert platform.count() == 2


def test_500_is_retried_twice(platform):
    platform.script = [500, 502]
    assert post(platform).status_code == 200
    assert platform.count() == 3


def test_gives_up_after_retries(platform):
    platform.script = [500, 500, 500]
    assert post(platform, retries=2).status_code == 500
    assert platform.count() == 3, "retries=2 一共该打 3 次"


def test_retry_after_header_is_honoured(platform):
    platform.script = [429]
    platform.retry_after = "0.5"
    started = time.monotonic()
    assert post(platform).status_code == 200
    assert time.monotonic() - started >= 0.5


def test_timeout_is_retried(platform):
    with pytest.raises(requests.Timeout):
        post(platform, path="slow", timeout=0.15, retries=1)
    assert platform.count() == 2


def test_backoff_grows_and_carries_jitter():
    # 每一档都比上一档的*最小*值大，所以带抖动也仍然单调
    plain = [net.backoff_delay(attempt) for attempt in range(4)]
    assert all(b > a for a, b in zip(plain, plain[1:])), "退避必须递增"

    samples = [net.backoff_delay(0) for _ in range(40)]
    assert len({round(value, 3) for value in samples}) > 1, "没有抖动，所有线程会同时醒来重撞"
    assert all(1.0 <= value <= 1.4 for value in samples)


def test_status_codes_are_classified():
    assert not net.is_retryable(401) and not net.is_retryable(400)
    assert not net.is_retryable(404) and not net.is_retryable(402)
    assert net.is_retryable(429) and net.is_retryable(500) and net.is_retryable(503)
