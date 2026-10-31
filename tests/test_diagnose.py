"""站点诊断：.cn 与 .com 的 Key 不通用，这件事必须由程序说出来。"""

import pytest

from rag import diagnose

CN = "https://api.siliconflow.cn/v1"
COM = "https://api.siliconflow.com/v1"


class FakeResponse:
    def __init__(self, status=200, payload=None):
        self.status_code = status
        self._payload = payload or {}

    def json(self):
        return self._payload


@pytest.fixture(autouse=True)
def clear_cache():
    diagnose.reset_cache()
    yield
    diagnose.reset_cache()


def test_site_detection():
    assert diagnose.site_of(CN) == "cn"
    assert diagnose.site_of(COM) == "com"
    assert diagnose.site_of("http://127.0.0.1:8000/v1") == ""


def test_mismatch_when_key_works_on_the_other_site(monkeypatch):
    calls = []

    def fake_get(url, headers=None, timeout=None):
        calls.append(url)
        return FakeResponse(200, {"data": []})

    monkeypatch.setattr(diagnose.requests, "get", fake_get)
    hint = diagnose.key_site_mismatch(CN, "sk-abc123pwxk")

    assert calls and calls[0].startswith(COM), "应该拿同一个 Key 去试另一个站点"
    assert "api.siliconflow.com" in hint and "不通用" in hint
    assert "SK-abc" not in hint, "提示里不能出现完整 Key"


def test_no_mismatch_when_both_sites_reject(monkeypatch):
    monkeypatch.setattr(
        diagnose.requests, "get",
        lambda url, headers=None, timeout=None: FakeResponse(401),
    )
    assert diagnose.key_site_mismatch(CN, "sk-dead") == ""
    message = diagnose.explain_401(CN, "sk-dead")
    assert "重新生成" in message


def test_balance_only_account_is_reported_as_such(monkeypatch):
    monkeypatch.setattr(
        diagnose.requests, "get",
        lambda url, headers=None, timeout=None: FakeResponse(402),
    )
    hint = diagnose.key_site_mismatch(CN, "sk-abc123pwxk")
    assert "余额不足" in hint and "不通用" in hint


def test_custom_host_never_probes(monkeypatch):
    def boom(*args, **kwargs):  # 不该被调用
        raise AssertionError("自定义地址不该偷偷发请求")

    monkeypatch.setattr(diagnose.requests, "get", boom)
    assert diagnose.key_site_mismatch("http://127.0.0.1:9999/v1", "sk-abc") == ""


def test_network_failure_is_silent(monkeypatch):
    def raise_error(*args, **kwargs):
        raise diagnose.requests.ConnectionError("boom")

    monkeypatch.setattr(diagnose.requests, "get", raise_error)
    assert diagnose.key_site_mismatch(CN, "sk-abc") == ""
    assert diagnose.model_alternatives(CN, "sk-abc", "embed") == []


def test_mismatch_is_cached(monkeypatch):
    calls = []

    def fake_get(url, headers=None, timeout=None):
        calls.append(url)
        return FakeResponse(200, {"data": []})

    monkeypatch.setattr(diagnose.requests, "get", fake_get)
    diagnose.key_site_mismatch(CN, "sk-abc123pwxk")
    diagnose.key_site_mismatch(CN, "sk-abc123pwxk")
    diagnose.key_site_mismatch(CN, "sk-abc123pwxk")
    assert len(calls) == 1, "同一次会话里不该反复探测"


def test_model_alternatives_are_filtered_by_kind(monkeypatch):
    catalog = {
        "data": [
            {"id": "deepseek-ai/DeepSeek-V3.2"},
            {"id": "Qwen/Qwen3-Embedding-8B"},
            {"id": "Qwen/Qwen3-Embedding-0.6B"},
            {"id": "Qwen/Qwen3-Reranker-8B"},
            {"id": "Qwen/Qwen3-VL-32B-Instruct"},
        ]
    }
    monkeypatch.setattr(
        diagnose.requests, "get",
        lambda url, headers=None, timeout=None: FakeResponse(200, catalog),
    )
    assert diagnose.model_alternatives(COM, "sk-abc", "embed") == [
        "Qwen/Qwen3-Embedding-8B",
        "Qwen/Qwen3-Embedding-0.6B",
    ]
    assert diagnose.model_alternatives(COM, "sk-abc", "rerank") == ["Qwen/Qwen3-Reranker-8B"]
    assert "Qwen/Qwen3-VL-32B-Instruct" in diagnose.model_alternatives(COM, "sk-abc", "vision")


def test_explain_404_lists_what_exists(monkeypatch):
    catalog = {"data": [{"id": "Qwen/Qwen3-Embedding-8B"}]}
    monkeypatch.setattr(
        diagnose.requests, "get",
        lambda url, headers=None, timeout=None: FakeResponse(200, catalog),
    )
    message = diagnose.explain_404(COM, "sk-abc", "BAAI/bge-m3", "embed")
    assert "BAAI/bge-m3" in message and "Qwen/Qwen3-Embedding-8B" in message
