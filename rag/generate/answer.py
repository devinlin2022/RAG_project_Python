"""Page-scoped question answering shared by the UI and the API."""

from rag.config import settings
from rag.generate.llm import chat
from rag.generate.prompt import build_context, build_messages


class PageAnswerError(ValueError):
    """Invalid input; message is safe to show."""


def select_pages(doc: dict, pages) -> tuple[list[dict], list[int]]:
    """Filter pages, ignoring out-of-range numbers."""
    available = {p["page"] for p in doc["pages"]}
    wanted = sorted({int(p) for p in pages if int(p) in available})
    return [p for p in doc["pages"] if p["page"] in wanted], wanted


def answer_by_pages(doc: dict, pages, question: str) -> dict:
    """Answer using only the given pages."""
    if not question or not question.strip():
        raise PageAnswerError("问题不能为空")

    chosen, picked = select_pages(doc, pages)
    if not picked:
        raise PageAnswerError("请至少选择一页；所选页码也需在文档范围内")

    context = build_context(chosen)
    if len(context) > settings.max_context_chars:
        raise PageAnswerError(
            f"所选 {len(picked)} 页共 {len(context)} 字符，超过上限 "
            f"{settings.max_context_chars}，请减少页数"
        )

    answer, usage = chat(build_messages(chosen, question))
    return {
        "answer": answer,
        "pages": picked,
        "chars": len(context),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
    }
