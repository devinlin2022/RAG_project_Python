"""Recursive character splitter, kept as the control arm."""

from rag.chunk.types import Chunk

SEPARATORS = ["\n\n", "\n", "。", "；", "！", "？", "，", " "]


def _split_once(text: str, separator: str) -> list[str]:
    """Split keeping the separator attached to the left piece."""
    if not separator:
        return [text]
    raw = text.split(separator)
    pieces = [piece + separator for piece in raw[:-1]]
    if raw[-1]:
        pieces.append(raw[-1])
    return [p for p in pieces if p]


def _recursive(text: str, size: int, separators: list[str]) -> list[str]:
    if len(text) <= size:
        return [text] if text.strip() else []

    for index, separator in enumerate(separators):
        if separator and separator in text:
            pieces: list[str] = []
            for piece in _split_once(text, separator):
                if len(piece) <= size:
                    pieces.append(piece)
                else:
                    pieces.extend(_recursive(piece, size, separators[index + 1 :]))
            return pieces

    # No separator works: hard cut, avoiding a split in the middle of a word when
    # a space is nearby.
    out: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            space = text.rfind(" ", start, end)
            if space > start + size // 2:
                end = space + 1
        out.append(text[start:end])
        start = end
    return [p for p in out if p.strip()]


def _merge(pieces: list[str], size: int, overlap: int) -> list[str]:
    chunks: list[str] = []
    current = ""
    for piece in pieces:
        if not current:
            current = piece
        elif len(current) + len(piece) <= size:
            current += piece
        else:
            chunks.append(current)
            tail = current[-overlap:] if overlap > 0 else ""
            current = (tail + piece) if tail else piece
    if current.strip():
        chunks.append(current)
    return chunks


def split_recursive(text: str, size: int, overlap: int = 0) -> list[str]:
    """Split plain text into pieces of at most `size` chars."""
    if not text.strip():
        return []
    size = max(size, 80)
    overlap = max(0, min(overlap, size // 2))
    return _merge(_recursive(text, size, SEPARATORS), size, overlap)


def chunk_recursive(
    blocks: list[tuple[int, str]],
    size: int,
    overlap: int,
    doc_id: str = "",
) -> list[Chunk]:
    """Chunk a document as one continuous stream, ignoring structure."""
    if not blocks:
        return []
    pages = [page for page, _ in blocks]
    text = "\n".join(block for _, block in blocks)
    pieces = split_recursive(text, size, overlap)

    chunks: list[Chunk] = []
    cursor = 0
    for index, piece in enumerate(pieces):
        # Approximate page attribution by character offset; cheap and adequate.
        offset = text.find(piece.strip()[:60], cursor)
        if offset < 0:
            offset = cursor
        ratio = offset / max(len(text), 1)
        page = pages[min(int(ratio * len(pages)), len(pages) - 1)]
        cursor = max(offset, cursor)
        chunks.append(
            Chunk(
                idx=index,
                text=piece.strip(),
                page_from=page,
                page_to=page,
                kind="text",
                strategy="recursive",
            )
        )
    return chunks
