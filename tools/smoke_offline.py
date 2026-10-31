"""端到端自检：不用真 Key，也能把整条链路跑一遍。

    python -m tools.smoke_offline

它做的事和真人操作一致：吃一份示例 PDF → 解析 → 建索引 → 提问（流式）→
自动出题 → 两种模式各跑一次评测，每一步都断言，最后再把平台换成"一直 401"
重跑一次，确认失败时只打一次请求、给的是能照着做的话。

意义在于：平台那边到底是 Key 失效、余额不足还是限流，本地分不清，但**我们的
处理对不对**可以在本地分清。Colab 上出了问题，先跑这个，能排除掉整整一半可能。
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data" / "samples" / "sample-3pages.pdf"

PASS, FAIL = "✓", "✗"
_failures: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  {PASS if ok else FAIL} {label}{('　' + detail) if detail else ''}")
    if not ok:
        _failures.append(label)


# 节流要量的是"客户端什么时候把请求发出去"。服务端记录的到达时间不行：
# 假平台是本地线程，几个处理线程会被成批唤醒，20ms 的发送间隔在到达端能压成 3ms，
# 于是这条断言会时好时坏——坏的时候不是节流坏了，是尺子坏了。
_SEND_STAMPS: list[float] = []


def install_send_spy() -> None:
    import rag.net as net

    original = net.requests.post

    def spy(*args, **kwargs):
        _SEND_STAMPS.append(time.monotonic())
        return original(*args, **kwargs)

    net.requests.post = spy


def send_gaps() -> list[float]:
    stamps = sorted(_SEND_STAMPS)
    return [round(b - a, 3) for a, b in zip(stamps, stamps[1:])]


def _configure(base_url: str, db_name: str) -> None:
    """必须在 import rag.config 之前设置：配置是模块级单例，导入即冻结。"""
    os.environ["SILICONFLOW_BASE_URL"] = base_url
    os.environ["SILICONFLOW_API_KEY"] = "sk-offline-smoke-test"
    os.environ["DB_FILENAME"] = db_name
    os.environ["OCR_ENABLED"] = "false"          # 扫描页 OCR 不在这次验证范围内
    os.environ["RERANK_ENABLED"] = "true"
    os.environ["HTTP_MIN_INTERVAL"] = "0.02"
    os.environ["HTTP_RETRIES"] = "2"
    os.environ["LOG_JSON"] = "false"
    os.environ["EMBED_CONCURRENCY"] = "2"
    os.environ["TEMPERATURE"] = "0"


def run_happy_path(platform) -> None:
    from rag.config import settings
    from app.state import get_service
    from rag.service import ServiceError  # noqa: F401  (确认异常类型可导入)

    print("\n[1] 正常链路")
    service = get_service()
    assert service.library.stats()["chunks"] == 0, "临时库应当是空的"

    results = service.ingest_files([str(SAMPLE)])
    check("解析示例 PDF", results[0].get("ok", False), results[0].get("error", ""))
    doc_id = results[0]["doc"]["id"]
    print(f"      {results[0]['doc']['filename']}：{results[0]['doc']['page_count']} 页 / "
          f"{results[0]['doc']['chars']} 字符")

    info = service.index(doc_id, force=True)
    check("建索引（向量 + BM25）", info["chunks"] > 0 and info["dim"] > 0,
          f"{info['chunks']} 块 / {info['dim']} 维 / 向量化 {info['embed_ms']:.0f}ms")
    check("向量化没有退化成关键词", not info["embed_error"], info["embed_error"])

    # ── 流式问答 ──
    started = time.perf_counter()
    pieces: list[str] = []
    done: dict = {}
    for event in service.ask_stream("付款账期最长是多少天？", mode="fast"):
        if isinstance(event, tuple):      # (阶段名, 中间结果) —— 流水线事件，不关心
            continue
        stage = event.get("stage")
        if stage == "delta":
            pieces.append(event.get("text") or "")
        elif stage == "done":
            done = event.get("result") or {}
        elif stage == "error":
            check("流式问答", False, str(event.get("error")))
            return

    answer = "".join(pieces)
    elapsed = (time.perf_counter() - started) * 1000
    check("流式问答返回内容", bool(answer.strip()), f"{len(answer)} 字 / {elapsed:.0f}ms")
    check("带回了引用来源", len(done.get("sources") or []) > 0,
          f"{len(done.get('sources') or [])} 条")
    usage = done.get("usage") or {}
    check("记录了 token 用量", bool(usage.get("total_tokens")), str(usage.get("total_tokens")))

    # ── 自动出题 ──
    print("\n[2] 自动出题")
    bundle = service.generate_questions(doc_ids=[doc_id], limit=4)
    questions = bundle["questions"]
    check("生成了带页码的题目", len(questions) > 0, f"{len(questions)} 道")
    check("每道题都有页码标注", all(item["pages"] and isinstance(item["pages"][0], int)
                                    for item in questions))
    check("题目没有失败", bundle["failed"] == 0, f"失败 {bundle['failed']} 个块")
    for item in questions[:3]:
        print(f"      p.{item['pages'][0]}　{item['q']}")

    # ── 两模式评测 ──
    print("\n[3] 两种模式评测（只检索，不生成）")
    for mode in ("fast", "enhanced"):
        result = service.evaluate(top_k=5, mode=mode, doc_ids=[doc_id], questions=questions)
        summary = result["summary"]
        check(f"{mode} 评测跑通", bool(result["rows"]),
              f"hit@{summary['k']} {summary['hit']:.0%} ｜ "
              f"MRR {summary['mrr']:.3f} ｜ {summary['questions']} 题")

    # ── 节流 ──
    print("\n[4] 请求节流")
    gaps = send_gaps()
    interval = settings.http_min_interval
    if gaps:
        check("相邻请求被拉开", min(gaps) >= interval * 0.9,
              f"最小间隔 {min(gaps) * 1000:.0f}ms（设定 {interval * 1000:.0f}ms，"
              f"共 {len(gaps) + 1} 次请求）")

    service.delete(doc_id)


def run_auth_failure(platform) -> None:
    """把平台切成永久 401，确认我们只打一次、并且说人话。"""
    from rag.config import settings
    from rag.generate import llm

    print("\n[5] Key 失效时的表现")
    platform.reset()
    platform.script("chat", [401] * 10)
    settings.llm_base_url = f"http://127.0.0.1:{platform.port}/v1"

    started = time.perf_counter()
    try:
        llm.chat([{"role": "user", "content": "hi"}])
        check("401 抛出异常", False, "居然没抛")
    except llm.LLMError as exc:
        message = str(exc)
        cost = (time.perf_counter() - started) * 1000
        check("提到了 401", "401" in message)
        check("指出了 Key 尾号", "尾号" in message)
        check("说清四条通道共用一个 Key", "共用" in message)
        check("只请求了一次", platform.hits("chat") == 1, f"{platform.hits('chat')} 次 / {cost:.0f}ms")

    print("\n[6] 429 之后能自愈")
    platform.reset()
    platform.script("chat", [429, 429])
    answer, _usage = llm.chat([{"role": "user", "content": "hi"}])
    check("重试后拿到答案", bool(answer.strip()), f"{platform.hits('chat')} 次请求")


def main() -> int:
    from tools.fake_platform import FakePlatform

    print("=" * 66)
    print("  离线自检：假平台代替硅基流动，验证链路与失败处理")
    print("=" * 66)

    platform = FakePlatform()
    base = platform.start()
    print(f"  假平台：{base}")

    db_name = f"smoke-{os.getpid()}.sqlite"
    db_path = ROOT / "data" / db_name
    _configure(base, db_name)
    install_send_spy()

    try:
        run_happy_path(platform)
        run_auth_failure(platform)
    finally:
        platform.stop()
        for path in (db_path, Path(str(db_path) + "-wal"), Path(str(db_path) + "-shm")):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass

    print("\n" + "=" * 66)
    if _failures:
        print(f"  {len(_failures)} 项没过：" + "、".join(_failures))
        return 1
    print("  全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
