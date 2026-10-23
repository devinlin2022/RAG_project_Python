"""Profile a document and recommend a chunking strategy in milliseconds.

No model call, no embedding: the recommender only reads the text that has already
been extracted. That is what keeps the recommendation cheap enough to run on every
upload, which a model-based router could not do.

The rules encode one opinion, stated openly: if a document has real numbering
(章/条 system), structure-first chunking wins, because a clause split in half is
worthless as evidence no matter how semantically coherent the halves look.
"""

from dataclasses import asdict, dataclass, field

from rag.chunk.semantic import split_sentences
from rag.chunk.structure import heading_stats

_MODALITY = ("应当", "不得", "可以", "有权", "须", "应", "视为", "承担", "约定", "负责")
_STRATEGIES = ("hybrid", "structured", "semantic", "recursive")


@dataclass
class DocProfile:
    chars: int
    lines: int
    pages: int
    heading_lines: int
    heading_ratio: float
    max_depth: int
    article_count: int
    styles: dict
    modality_density: float
    avg_sentence_len: int
    p90_paragraph_len: int
    table_page_ratio: float
    figure_page_ratio: float
    structure_score: float
    strategy: str
    chunk_size: int
    overlap: int
    confidence: str
    reasons: list[str] = field(default_factory=list)
    alternatives: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _percentile(values: list[int], pct: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = min(int(len(ordered) * pct / 100), len(ordered) - 1)
    return ordered[index]


def profile_document(
    pairs: list[tuple[int, object]],
    pages: int = 0,
    table_pages: int = 0,
    figure_pages: int = 0,
) -> DocProfile:
    """Census the text and pick a strategy with written-down reasons."""
    text = "\n".join(getattr(block, "text", "") for _, block in pairs)
    lines = [line for line in text.splitlines() if line.strip()]
    chars = len(text)

    stats = heading_stats(text)
    sentences = split_sentences(text)
    paragraphs = [p for p in text.split("\n") if p.strip()]

    modality_hits = sum(1 for s in sentences if any(word in s for word in _MODALITY))
    modality = round(modality_hits / len(sentences), 4) if sentences else 0.0
    avg_sentence = round(len(text) / len(sentences)) if sentences else 0
    p90_para = _percentile([len(p) for p in paragraphs], 90)

    table_ratio = round(table_pages / pages, 3) if pages else 0.0
    figure_ratio = round(figure_pages / pages, 3) if pages else 0.0

    structure_score = round(
        min(stats.ratio / 0.05, 1.0) * 0.55 + min(stats.article_count / 12, 1.0) * 0.45,
        4,
    )

    reasons: list[str] = []
    legal_like = stats.article_count >= 5 or (stats.ratio >= 0.015 and stats.max_depth >= 2)

    if legal_like:
        strategy = "hybrid"
        reasons.append(
            f"检测到 {stats.max_depth} 级标题、{stats.article_count} 条条款，属条款型文档"
        )
        reasons.append("结构优先切分：条款不跨块，检索命中即完整证据")
        reasons.append("超长条款再按语义断点切分，兼顾长度上限")
        confidence = "高" if stats.article_count >= 8 else "中"
    elif stats.ratio >= 0.01 and stats.max_depth >= 2:
        strategy = "structured"
        reasons.append(f"有 {stats.heading_lines} 行标题但条款编号稀疏，按标题切分即可")
        reasons.append("不做语义切分，省掉一次 Embedding 调用")
        confidence = "中"
    elif chars >= 3000 and len(sentences) >= 20:
        strategy = "semantic"
        reasons.append("未检出稳定编号体系，按语义断点切分")
        reasons.append("需要 Embedding 调用，耗时随句数增长")
        confidence = "中"
    else:
        strategy = "recursive"
        reasons.append("文档过短或结构信号不足，递归切分最省时")
        confidence = "低"

    size = 500
    if stats.article_count >= 10 and p90_para and p90_para < 320:
        size = 400
        reasons.append(f"条款段落 90 分位仅 {p90_para} 字，块长下调到 400")
    if table_ratio >= 0.2:
        size = max(size, 700)
        reasons.append(f"表格页占 {table_ratio:.0%}，块长上调以保证表格不被切碎")

    size = max(300, min(size, 900))
    overlap = 0 if strategy in ("hybrid", "structured") else (size // 6 if strategy == "semantic" else 60)
    if overlap == 0 and strategy in ("hybrid", "structured"):
        reasons.append("结构切分不需要重叠：边界本身就是语义边界，重叠只会引入噪声")

    if modality >= 0.3:
        reasons.append(f"规范性用语占比 {modality:.0%}，确认属制度/合同语体")

    alternatives = [s for s in _STRATEGIES if s != strategy]
    return DocProfile(
        chars=chars,
        lines=len(lines),
        pages=pages,
        heading_lines=stats.heading_lines,
        heading_ratio=round(stats.ratio, 4),
        max_depth=stats.max_depth,
        article_count=stats.article_count,
        styles=stats.styles,
        modality_density=modality,
        avg_sentence_len=avg_sentence,
        p90_paragraph_len=p90_para,
        table_page_ratio=table_ratio,
        figure_page_ratio=figure_ratio,
        structure_score=structure_score,
        strategy=strategy,
        chunk_size=size,
        overlap=overlap,
        confidence=confidence,
        reasons=reasons,
        alternatives=alternatives,
    )
