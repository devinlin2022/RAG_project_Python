"""Tokenisation shared by BM25 and any lexical score.

Two rules matter more than the segmenter itself:

- Latin / code tokens stay whole. Splitting "INV-2048" into "inv" and "2048" is how
  BM25 stops matching the identifiers that appear in tables.
- Chinese falls back to character bigrams when jieba is unavailable, so the
  lexical rail never depends on an optional package.
"""

import re

_LATIN = re.compile(r"[A-Za-z][A-Za-z0-9]*(?:[&./_-][A-Za-z0-9]+)*")
_NUMBER = re.compile(r"\d+(?:[.,]\d+)*%?")
_CJK = re.compile(r"[\u4e00-\u9fff]+")

# Function words and single-character fillers: high df, no discriminative power.
_STOP = frozenset(
    "的 了 和 与 及 或 在 是 为 对 中 上 下 我们 你们 他们 本 该 这 那 之 其 等 也 就 都 而 并 由 从 到".split()
)

try:  # optional, purely a quality upgrade
    import jieba

    jieba.setLogLevel(60)
    _HAS_JIEBA = True
except Exception:  # pragma: no cover - depends on install
    _HAS_JIEBA = False


def _cjk_tokens(run: str) -> list[str]:
    if _HAS_JIEBA:
        words = [w.strip() for w in jieba.cut(run) if w.strip()]
        tokens = [w for w in words if len(w) > 1 and w not in _STOP]
        if tokens:
            return tokens
    # Fallback: overlapping bigrams keep recall acceptable without a dictionary.
    if len(run) == 1:
        return [] if run in _STOP else [run]
    return [run[i : i + 2] for i in range(len(run) - 1)]


def tokenize(text: str) -> list[str]:
    """Lowercased tokens: Latin words, numbers, and Chinese words/bigrams."""
    if not text:
        return []
    tokens: list[str] = []
    for match in _LATIN.finditer(text):
        tokens.append(match.group(0).lower())
    for match in _NUMBER.finditer(text):
        tokens.append(match.group(0).lower())
    for match in _CJK.finditer(text):
        tokens.extend(_cjk_tokens(match.group(0)))
    return tokens


def has_jieba() -> bool:
    return _HAS_JIEBA
