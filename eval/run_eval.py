"""Golden-set runner: retrieval metrics for every strategy, citations optional.

`python -m eval.run_eval --doc demo-policy --top-k 5`
"""

import argparse
import json
import sys
import time
from pathlib import Path

from eval.metrics import citation_grounding, is_abstain, rank_of_first_hit

GOLDEN = Path(__file__).with_name("golden.jsonl")


def load_golden(path: str | Path | None = None) -> list[dict]:
    target = Path(path) if path else GOLDEN
    if not target.exists():
        return []
    items = []
    for line in target.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            items.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return items


def run_questions(
    service,
    questions: list[dict],
    doc_pattern: str = "",
    top_k: int = 5,
    strategy: str = "rrf",
    with_generation: bool = False,
    limit: int | None = None,
    doc_ids: list[str] | None = None,
) -> tuple[list[dict], int]:
    """Run the golden set. Returns (rows, skipped_count)."""
    from rag.retrieve.retriever import rerank_hits

    known = {doc["id"]: doc["filename"].lower() for doc in service.list_documents()}
    picked = doc_ids if doc_ids is not None else list(known)
    rows: list[dict] = []
    skipped = 0

    for question in questions[: limit or len(questions)]:
        target = (question.get("doc") or "").strip().lower()
        if target:
            scoped = [doc_id for doc_id in picked if target in known.get(doc_id, "")]
            if not scoped:
                skipped += 1
                continue
        else:
            scoped = picked
        if not scoped:
            skipped += 1
            continue

        started = time.perf_counter()
        result = service.retrieve(question["q"], scoped, top_k=top_k, strategy=strategy)
        spans = [(hit.record.page_from, hit.record.page_to) for hit in result["hits"]]
        pages = [first for first, _ in spans]
        gold = [int(p) for p in question.get("pages", [])]
        rank = rank_of_first_hit(spans, gold, top_k)
        row = {
            "id": question.get("id", ""),
            "q": question["q"],
            "doc": scoped[0],
            "gold": gold,
            "pages": pages,
            "hit": 1.0 if rank else 0.0,
            "rr": 1.0 / rank if rank else 0.0,
            "answerable": bool(question.get("answerable", True)),
            "ms": round((time.perf_counter() - started) * 1000, 1),
            "reranked": bool(result["rerank"]["used"]),
        }

        if with_generation:
            try:
                answer = service.ask(question["q"], scoped, top_k=top_k, strategy=strategy)
                valid, total = citation_grounding(answer["answer"], len(answer["sources"]))
                row.update(
                    answer=answer["answer"],
                    abstained=is_abstain(answer["answer"]),
                    cite_valid=valid,
                    cite_total=total,
                    gen_ms=answer["timings"]["total_ms"],
                )
            except Exception as exc:  # a failed call must not abort the run
                row.update(answer=f"（生成失败：{exc}）", abstained=False, cite_valid=0, cite_total=0)
        rows.append(row)

    return rows, skipped


def format_markdown(rows: list[dict], summary: dict, skipped: int, meta: dict) -> str:
    lines = [
        "# 评测报告",
        "",
        f"- 检索策略：`{meta.get('strategy')}`　｜　hit@{summary['k']}　｜　"
        f"问题 {summary['questions']} 条（跳过 {skipped} 条）",
        f"- 模型：`{meta.get('model', '—')}`　｜　索引块数 {meta.get('chunks', 0)}",
        "",
        "| 指标 | 值 |",
        "| --- | --- |",
        f"| hit@{summary['k']} | {summary['hit']:.1%} |",
        f"| MRR | {summary['mrr']:.3f} |",
    ]
    if summary["grounding"] is not None:
        lines.append(f"| 引用有效率 | {summary['grounding']:.1%}（{summary['citation_total']} 处引用） |")
    if summary["abstain_accuracy"] is not None:
        lines.append(f"| 拒答准确率 | {summary['abstain_accuracy']:.1%} |")
    lines += ["", "| # | 问题 | 命中页码 | gold | hit | RR |", "| --- | --- | --- | --- | --- | --- |"]
    for row in rows:
        lines.append(
            f"| {row['id']} | {row['q'][:40]} | {row['pages'][:5]} | {row['gold']} | "
            f"{'✓' if row['hit'] else '✗'} | {row['rr']:.2f} |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the golden set against the local index")
    parser.add_argument("--doc", default="", help="filename substring, e.g. demo-policy")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--strategy", default="rrf", choices=["rrf", "dense", "sparse"])
    parser.add_argument("--generate", action="store_true", help="also call the LLM for citation metrics")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", default="eval/report.md")
    args = parser.parse_args(argv)

    from rag.service import Service

    service = Service()
    if not service.library.records:
        print("索引为空：请先在页面上传 PDF 并建立索引，或运行 python -m rag.cli ingest <path>")
        return 1

    result = service.evaluate(
        doc_pattern=args.doc,
        top_k=args.top_k,
        strategy=args.strategy,
        with_generation=args.generate,
        limit=args.limit,
    )
    summary = result["summary"]
    markdown = format_markdown(
        result["rows"],
        summary,
        result["skipped"],
        {
            "strategy": args.strategy,
            "model": service.system_info()["llm"]["model"],
            "chunks": service.library.stats()["chunks"],
        },
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(markdown, encoding="utf-8")
    print(markdown)
    print(f"\n写入 {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
