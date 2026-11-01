"""Blocking loading dialog: what the browser receives."""

from app import loading


def test_vendor_library_is_present_and_inlined():
    assert loading._LIB, "SweetAlert2 must ship with the repo, not come from a CDN"
    assert "sweetalert2" in loading._LIB.lower()
    assert "SweetAlert2" in loading.HEAD
    assert "FBLoad" in loading.HEAD


def test_open_js_carries_title_and_phases():
    script = loading.open_js("正在解析", "先读版面", 0, [(0, "第一步"), (4, "第二步")])
    assert script.startswith("(...args) =>")
    assert "return args" in script, "js 必须回传输入，否则 Gradio 会把所有输入置空"
    assert "正在解析" in script and "第一步" in script
    assert '"delay": 0' in script


def test_open_js_mode_switches_on_the_page_attribute():
    script = loading.open_js_mode(
        loading.payload("快速"), loading.payload("增强", phases=[(0, "精排")])
    )
    assert "data-fb-mode" in script
    assert "快速" in script and "增强" in script and "精排" in script


def test_close_is_safe_when_the_dialog_never_opened():
    assert "FBLoad" in loading.CLOSE_JS
    assert "close()" in loading.CLOSE_JS


def test_head_stays_below_a_megabyte():
    assert len(loading.HEAD.encode("utf-8")) < 200_000
