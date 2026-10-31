"""Build an evaluation set from the indexed chunks, one question per chunk."""

from __future__ import annotations

import json
import re
from typing import Callable, Iterable

from rag.generate.llm import LLMError, chat

PROMPT = """你在为一个检索系统出题。下面是一份文档里的一个片段，请据此写 {count} 个真实用户会问的问题。

要求：
- 每个问题都必须只用这个片段就能回答
- 用中文，像一个业务同事在提问，不要写"请根据以下内容"
- 专有名词、数字、部门名要和片段里完全一致
- 不要问"这一页讲了什么""这份文档的主要内容"这类范围过大的问题
- 问题之间不要重复

片段（第 {page} 页，来自《{filename}》）：
\"\"\"
{text}
\"\"\"

只输出 JSON 数组，不要解释：
[{{"q": "问题", "answer": "片段中对应的答案"}}]
"""

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def _parse(text: str) -> list[dict]:
    body = text.strip()
    fenced = _FENCE.search(body)
    if fenced:
        body = fenced.group(1).strip()
    start = min((i for i in (body.find("["), body.find("{")) if i >= 0), default=-1)
    if start > 0:
        body = body[start:]

    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        end = body.rfind("]")
        if end < 0:
            return []
        try:
            data = json.loads(body[: end + 1])
        except json.JSONDecodeError:
            return []

    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        return []

    items = []
    for row in data:
        if not isinstance(row, dict):
            continue
        question = str(row.get("q") or row.get("question") or "").strip()
        if len(question) < 6:
            continue
        items.append({"q": question, "answer": str(row.get("answer") or row.get("ground_truth") or "").strip()})
    return items


def sample_records(records: Iterable, limit: int) -> list:
    buckets: dict[tuple[str, int], list] = {}
    for record in records:
        if not (record.text or "").strip():
            continue
        buckets.setdefault((record.doc_id, record.page_from), []).append(record)

    picked = [group[0] for _, group in sorted(buckets.items())]
    if len(picked) < limit:
        chosen = {id(item) for item in picked}
        for group in buckets.values():
            for record in group[1:]:
                if id(record) not in chosen:
                    picked.append(record)
                    if len(picked) >= limit:
                        break
            if len(picked) >= limit:
                break
    return picked[:limit]


def generate_questions(
    records: Iterable,
    limit: int = 10,
    per_chunk: int = 1,
    on_progress: Callable[[int, int, str], None] | None = None,
) -> dict:
    samples = sample_records(records, limit)
    questions: list[dict] = []
    failed = 0

    for index, record in enumerate(samples, start=1):
        if on_progress:
            on_progress(index, len(samples), record.filename)
        prompt = PROMPT.format(
            count=per_chunk,
            page=record.page_from,
            filename=record.filename,
            text=(record.text or "")[:1200],
        )
        try:
            answer, _usage = chat([{"role": "user", "content": prompt}], temperature=0.3, max_tokens=400)
            items = _parse(answer)
        except LLMError:
            items = []
        except Exception:
            items = []

        if not items:
            failed += 1
            continue

        for item in items[:per_chunk]:
            questions.append(
                {
                    "id": f"auto-{len(questions) + 1:02d}",
                    "doc": record.filename,
                    "q": item["q"],
                    "pages": [int(record.page_from)],
                    "answerable": True,
                    "ground_truth": item["answer"],
                    "contexts": [(record.text or "")[:400]],
                    "source_chunk": record.idx,
                    "origin": "auto",
                }
            )

    return {"questions": questions, "failed": failed, "sampled": len(samples)}
