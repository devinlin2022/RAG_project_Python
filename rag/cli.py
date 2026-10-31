"""Headless pipeline: ingest, index and ask without the UI."""

import argparse
import sys

from rag.service import Service, ServiceError


def _service() -> Service:
    return Service()


def cmd_list(service: Service, args) -> int:
    rows = service.list_documents()
    if not rows:
        print("库为空：先运行 ingest")
        return 0
    print(f"{'ID':18s} {'页':>5s} {'字符':>8s} {'块':>5s}  {'策略':12s} 文件名")
    for row in rows:
        print(
            f"{row['id']:18s} {row['pages']:5d} {row['chars']:8d} {row['chunks']:5d}"
            f"  {row['strategy']:12s} {row['filename']}"
        )
    return 0


def cmd_ingest(service: Service, args) -> int:
    try:
        doc, existed = service.ingest(args.path)
    except ValueError as exc:
        print(f"解析失败：{exc}")
        return 1
    print(f"{'已存在' if existed else '已入库'} {doc['id']} {doc['filename']} "
          f"{doc['page_count']} 页 / {doc['chars']:,} 字符")
    if args.index:
        return cmd_index(service, argparse.Namespace(doc=doc["id"], force=False))
    return 0


def cmd_index(service: Service, args) -> int:
    targets = service.doc_ids_matching(args.doc)
    if not targets:
        print("没有匹配的文档")
        return 1
    for doc_id in targets:
        try:
            info = service.index(doc_id, force=args.force)
        except ServiceError as exc:
            print(f"{doc_id}: {exc}")
            continue
        state = "已存在，跳过" if info["skipped"] else f"{info['chunks']} 块 / {info['dim']} 维"
        print(f"{doc_id}: {state}（{info['strategy']}）")
        if info["embed_error"]:
            print(f"  Embedding 未生效：{info['embed_error']}")
    print("索引规模：", service.library.stats())
    return 0


def cmd_ask(service: Service, args) -> int:
    doc_ids = service.doc_ids_matching(args.doc) or None
    try:
        result = service.ask(args.question, doc_ids, top_k=args.top_k)
    except ServiceError as exc:
        print(f"失败：{exc}")
        return 1
    print(result["answer"])
    print()
    for item in result["sources"]:
        print(f"[{item['n']}] {item['filename']} p.{item['page']} · {item['kind']}")
    timings = result["timings"]
    print(
        f"\n检索 {timings.get('retrieve_ms', 0):.0f}ms · 精排 {timings.get('rerank_ms', 0):.0f}ms"
        f" · 生成 {timings.get('generate_ms', 0):.0f}ms"
        f" · 引用 {result['citations']['valid']}/{result['citations']['total']}"
    )
    return 0


def cmd_eval(service: Service, args) -> int:
    result = service.evaluate(
        doc_pattern=args.doc,
        top_k=args.top_k,
        strategy=args.strategy,
        with_generation=args.generate,
        limit=args.limit,
    )
    summary = result["summary"]
    if not summary["questions"]:
        print("没有可评测的问题（golden.jsonl 中的文档不在库里）")
        return 1
    print(
        f"hit@{summary['k']} {summary['hit']:.1%} · MRR {summary['mrr']:.3f} · "
        f"问题 {summary['questions']} 条 · 跳过 {result['skipped']} 条"
    )
    if summary["grounding"] is not None:
        print(f"引用有效率 {summary['grounding']:.1%}")
    if summary["abstain_accuracy"] is not None:
        print(f"拒答准确率 {summary['abstain_accuracy']:.1%}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rag.cli", description="rag-agent 命令行")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="列出库内文档")

    p_ingest = sub.add_parser("ingest", help="解析并入库一个 PDF")
    p_ingest.add_argument("path")
    p_ingest.add_argument("--index", action="store_true", help="入库后立即建索引")

    p_index = sub.add_parser("index", help="为文档建立向量/关键词索引")
    p_index.add_argument("--doc", default="", help="文件名包含的字符串，留空=全部")
    p_index.add_argument("--force", action="store_true")

    p_ask = sub.add_parser("ask", help="提问（返回答案、来源与耗时）")
    p_ask.add_argument("question")
    p_ask.add_argument("--doc", default="", help="限定文档，留空=全部")
    p_ask.add_argument("--top-k", type=int, default=6)

    p_eval = sub.add_parser("eval", help="运行 golden set")
    p_eval.add_argument("--doc", default="")
    p_eval.add_argument("--top-k", type=int, default=5)
    p_eval.add_argument("--strategy", default="rrf", choices=["rrf", "dense", "sparse"])
    p_eval.add_argument("--generate", action="store_true")
    p_eval.add_argument("--limit", type=int, default=None)

    args = parser.parse_args(argv)
    service = _service()
    handlers = {
        "list": cmd_list,
        "ingest": cmd_ingest,
        "index": cmd_index,
        "ask": cmd_ask,
        "eval": cmd_eval,
    }
    return handlers[args.cmd](service, args)


if __name__ == "__main__":
    sys.exit(main())
