"""Page routing and the incremental enhancement contract."""

from rag.ingest.enhance import Block, enhance, plan_pages


def _page(no, text="", scan=False, table=False, figure=False):
    return {
        "page": no,
        "text": text,
        "chars": len(text),
        "scan_like": scan,
        "table_like": table,
        "figure_like": figure,
        "image_cover": 0.9 if scan else 0.0,
        "vector_ops": 0,
        "tables": 1 if table else 0,
        "column_ratio": 0.0,
    }


def test_routes_follow_the_page_signals():
    plans = plan_pages({"pages": [_page(1, "正文"), _page(2, table=True), _page(3, scan=True)]})
    assert plans[0].routes == ["text"]
    assert plans[1].routes == ["text", "table"]
    assert plans[2].routes == ["ocr"]


def test_scan_page_is_skipped_without_ocr(offline):
    doc = {"pages": [_page(1, "", scan=True)], "path": "missing.pdf"}
    report = enhance(doc, limit_pages=5)
    assert report["api_calls"] == 0
    assert report["pages"][0].status == "skipped"
    # A skipped page must not be stored: otherwise the next run would treat it
    # as already enhanced and never retry it.
    assert 1 not in doc.get("blocks", {})


def test_cached_page_skips_the_model(offline):
    doc = {
        "pages": [_page(1, "", scan=True)],
        "path": "missing.pdf",
        "blocks": {1: [Block("ocr", "已识别的文字", "cached")]},
    }
    report = enhance(doc, limit_pages=5)
    assert report["api_calls"] == 0
    assert report["cached"] == 1
    assert doc["blocks"][1][0].text == "已识别的文字"


def test_force_reruns_cached_pages(offline):
    doc = {
        "pages": [_page(1, "", scan=True)],
        "path": "missing.pdf",
        "blocks": {1: [Block("ocr", "旧结果", "cached")]},
    }
    report = enhance(doc, limit_pages=5, force=True)
    assert report["cached"] == 0
    assert report["pages"][0].status == "skipped"


def test_text_pages_cost_nothing(offline):
    doc = {"pages": [_page(1, "普通正文，没有表格也没有图")], "path": "missing.pdf"}
    report = enhance(doc, limit_pages=5)
    assert report["api_calls"] == 0
    assert report["pages"][0].chars_after == doc["pages"][0]["chars"]
