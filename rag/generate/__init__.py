"""Answer generation: page-scoped QA, prompts, streaming chat."""

from rag.generate.answer import PageAnswerError, answer_by_pages, select_pages
from rag.generate.llm import LLMError, chat
from rag.generate.prompt import SYSTEM_PROMPT, build_context, build_messages

__all__ = [
    "LLMError",
    "PageAnswerError",
    "SYSTEM_PROMPT",
    "answer_by_pages",
    "build_context",
    "build_messages",
    "chat",
    "select_pages",
]
