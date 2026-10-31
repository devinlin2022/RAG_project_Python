"""Turn a follow-up question into a self-contained one."""

import logging
import re

log = logging.getLogger(__name__)

# Only these need a rewrite. Rewriting an already self-contained question costs
# one round trip and can drift away from what the user actually asked.
_ANAPHORA = re.compile(r"它|他|她|这个|那个|该|上述|前面|刚才|这些|那些|其中|此处|继续|再|还有|呢")

REWRITE_PROMPT = """把用户的最新问题改写成可以独立检索的问题。

规则：
- 结合对话历史补全省略的主语、对象和限定条件。
- 只输出改写后的问题本身，不要解释、不要引号、不要编号。
- 若最新问题本身已经完整，原样输出。
- 保留原文中的专有名词、编号、金额与比例。
- 疑问语气保持不变。"""


def needs_rewrite(question: str, history: list[dict] | None) -> bool:
    if not history:
        return False
    text = question.strip()
    if len(text) <= 6:
        return True
    return bool(_ANAPHORA.search(text))


def rewrite_question(
    question: str,
    history: list[dict] | None,
    chat_fn,
    turns: int = 3,
    enabled: bool = True,
) -> tuple[str, bool, str]:
    """Return (query used for retrieval, was_rewritten, error)."""
    if not enabled or not needs_rewrite(question, history):
        return question.strip(), False, ""

    transcript = "\n".join(
        f"{'用户' if turn.get('role') == 'user' else '助手'}：{turn.get('content', '')[:400]}"
        for turn in (history or [])[-turns * 2 :]
    )
    messages = [
        {"role": "system", "content": REWRITE_PROMPT},
        {
            "role": "user",
            "content": f"【对话历史】\n{transcript}\n\n【最新问题】\n{question.strip()}",
        },
    ]
    try:
        rewritten, _ = chat_fn(messages, temperature=0.0, max_tokens=120)
    except Exception as exc:  # rewrite is an optimisation, never a blocker
        log.warning("query rewrite failed: %s", exc)
        return question.strip(), False, str(exc)

    cleaned = rewritten.strip().strip("。\"'“”").splitlines()[0][:200] if rewritten.strip() else ""
    if not cleaned:
        return question.strip(), False, ""
    return cleaned, True, ""
