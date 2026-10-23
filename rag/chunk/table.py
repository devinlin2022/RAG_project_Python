"""Table splitting wrapper so the chunk package does not depend on pdfplumber."""

import logging

from rag.ingest.table import split_markdown_table

log = logging.getLogger(__name__)


def split_markdown_table_guard(block: str, max_chars: int) -> list[str]:
    """Split a Markdown table, repeating the header. Never raises."""
    try:
        return split_markdown_table(block, max_chars) or [block]
    except Exception as exc:
        log.warning("markdown table split failed: %s", exc)
        return [block]
