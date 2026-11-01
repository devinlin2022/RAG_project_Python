"""Blocking loading dialog: SweetAlert2, an elapsed clock and drifting trivia.

Long steps run behind a modal so a second click cannot start the same job again.
The library is vendored and inlined into <head> — no CDN, so Colab and offline
installs render the same dialog. Everything degrades to "no dialog" if the
script is missing: waiting is slow, never broken.
"""

import json
from pathlib import Path

VENDOR = Path(__file__).resolve().parent / "vendor" / "sweetalert2.min.js"

_LIB = VENDOR.read_text(encoding="utf-8") if VENDOR.exists() else ""

_CSS = """
.fb-swal-container { z-index: 100000 !important; }
.fb-swal-popup { border-radius: 16px !important; padding: 0 !important; overflow: hidden;
  max-width: 92vw; box-shadow: 0 24px 64px rgba(17, 24, 39, .22); }
.fb-load { padding: 24px 26px 18px; text-align: left; font-family: inherit; }
.fb-load-head { display: flex; align-items: center; gap: 14px; }
.fb-ring { width: 34px; height: 34px; border-radius: 50%; flex: 0 0 34px;
  border: 3px solid #dbeafe; border-top-color: #1a56db; animation: fb-spin .9s linear infinite; }
@keyframes fb-spin { to { transform: rotate(360deg); } }
.fb-load-title { font-size: 16px; font-weight: 600; color: #111827; line-height: 1.35; }
.fb-load-note { font-size: 12.5px; color: #6b7280; margin-top: 3px; }
.fb-load-clock { margin-left: auto; color: #1a56db; font-size: 20px; font-weight: 600;
  font-variant-numeric: tabular-nums; }
.fb-load-clock small { font-size: 11px; color: #9ca3af; margin-left: 3px; }
.fb-load-stage { margin-top: 14px; display: flex; align-items: center; min-height: 34px;
  padding: 8px 10px; border-radius: 8px; background: #f3f4f6; font-size: 13px; color: #374151; }
.fb-load-stage::before { content: ''; width: 6px; height: 6px; border-radius: 50%;
  background: #1a56db; margin-right: 8px; flex: 0 0 6px; }
.fb-dm { position: relative; height: 92px; margin-top: 12px; overflow: hidden;
  border: 1px dashed #e5e7eb; border-radius: 10px; background: #fbfdff;
  -webkit-mask-image: linear-gradient(90deg, transparent 0, #000 28px, #000 calc(100% - 56px), transparent 100%);
  mask-image: linear-gradient(90deg, transparent 0, #000 28px, #000 calc(100% - 56px), transparent 100%); }
.fb-dm-item { position: absolute; left: 100%; white-space: nowrap; font-size: 12.5px;
  color: #6b7280; animation: fb-dm-run linear forwards; animation-duration: 12s; }
@keyframes fb-dm-run {
  from { transform: translateX(0); }
  to { transform: translateX(calc(-1 * var(--fb-w, 460px) - 100%)); }
}
.fb-load-tip { margin-top: 12px; padding-top: 10px; border-top: 1px solid #f3f4f6;
  font-size: 12.5px; color: #4b5563; min-height: 40px; }
.fb-tip-tag { display: inline-block; margin-right: 8px; padding: 2px 6px; border-radius: 6px;
  background: #eff6ff; color: #1a56db; font-size: 11px; }
.fb-tip-a { color: #9ca3af; transition: opacity .2s; }
@media (prefers-reduced-motion: reduce) {
  .fb-ring { animation: none; }
  .fb-dm { display: none; }
}
"""

_SCRIPT = r"""
(function () {
  if (window.FBLoad) return;

  var TIPS = [
    ["为什么扫描件的同一行字会忽高忽低？", "进纸时纸张轻微倾斜，OCR 会按行重新拉直。"],
    ["表格变成纯文本后，为什么对不齐了？", "空格宽度不固定，所以这里把它转成 Markdown 表格。"],
    ["一句话被切成两半，检索会漏掉吗？", "相邻块留了重叠字数，就是防这种情况。"],
    ["为什么有的页一秒读完，有的一页要半分钟？", "前者有文本层，后者要交给视觉模型逐字认。"],
    ["文件第 3 页和正文第 3 页，是同一页吗？", "不一定，封面与目录常常不计入正文页码。"],
    ["文档里没写，模型为什么还会答？", "那是幻觉。这里的提示词要求它明确说没写。"],
    ["同一个问题问两遍，答案会变吗？", "温度大于 0 就会。这里设成 0.2，尽量稳住。"],
    ["编号、金额、缩写，谁更敏感？", "关键词检索。所以两路都跑，只比较名次。"],
    ["为什么先粗排 36 条，再精排？", "精排逐条打分很贵，先用便宜的方式把范围缩小。"],
    ["引用编号 [2] 指的是第 2 段吗？", "是第 2 个命中块，旁边写着它来自第几页。"],
    ["同一份文档，为什么两次命中页不同？", "切块边界或精排模型变了，命中就会挪动。"],
    ["整份 PPT 会被切成一个块吗？", "不会，一页一块；否则引用会写成 p.1-12。"]
  ];

  var state = { opens: 0, open: false, started: 0, timer: null, spawner: null,
                pending: null, phases: [], tip: -1, lane: 0 };

  function setText(id, value) {
    var node = document.getElementById(id);
    if (node && value != null) node.textContent = value;
  }

  function seconds() { return (Date.now() - state.started) / 1000; }

  function stageLabel(s) {
    var label = state.phases.length ? state.phases[0][1] : '正在处理';
    for (var i = 0; i < state.phases.length; i++) {
      if (s >= state.phases[i][0]) label = state.phases[i][1];
    }
    if (s > 90) return label + '（比平时久，可能碰到了限流，可以继续等）';
    return label;
  }

  function tick() {
    var s = seconds();
    setText('fb-load-sec', s < 10 ? s.toFixed(1) : String(Math.round(s)));
    setText('fb-load-stage', stageLabel(s));
    // Last-resort release: if a completion callback never fires, the overlay
    // must not trap the page forever.
    if (s > 300 && state.open) { state.opens = 0; window.FBLoad.close(); }
  }

  function spawn() {
    var box = document.getElementById('fb-dm');
    if (!box) return;
    box.style.setProperty('--fb-w', box.clientWidth + 'px');
    state.tip = (state.tip + 1) % TIPS.length;
    var item = TIPS[state.tip];
    var line = document.createElement('div');
    line.className = 'fb-dm-item';
    line.style.top = (6 + (state.lane % 3) * 30) + 'px';
    // The first line starts mid-track: an empty band for two seconds reads as
    // "nothing is happening" on exactly the waits people are impatient about.
    if (state.lane === 0) line.style.animationDelay = '-4.5s';
    line.textContent = item[0];
    box.appendChild(line);
    state.lane += 1;
    setTimeout(function () { if (line.parentNode) line.parentNode.removeChild(line); }, 14000);

    var answer = document.getElementById('fb-tip-a');
    if (answer) {
      answer.style.opacity = '0';
      setTimeout(function () { answer.textContent = item[1]; answer.style.opacity = '1'; }, 200);
    }
  }

  function body() {
    return '' +
      '<div class="fb-load">' +
      '  <div class="fb-load-head">' +
      '    <div class="fb-ring"></div>' +
      '    <div><div class="fb-load-title" id="fb-load-title">正在处理</div>' +
      '      <div class="fb-load-note" id="fb-load-note"></div></div>' +
      '    <div class="fb-load-clock"><span id="fb-load-sec">0.0</span><small>秒</small></div>' +
      '  </div>' +
      '  <div class="fb-load-stage" id="fb-load-stage">正在准备</div>' +
      '  <div class="fb-dm" id="fb-dm"></div>' +
      '  <div class="fb-load-tip"><span class="fb-tip-tag">小知识</span>' +
      '    <span class="fb-tip-a" id="fb-tip-a">文档读得越全，回答越稳；这一步没有缓存，每次都是现算。</span>' +
      '  </div>' +
      '</div>';
  }

  function open(opts) {
    if (state.open) return;
    state.open = true;
    state.started = Date.now();
    state.lane = 0;
    state.tip = -1;
    state.phases = opts.phases || [];
    if (typeof Swal === 'undefined') { state.open = false; return; }

    Swal.fire({
      html: body(),
      width: 520,
      padding: 0,
      background: '#ffffff',
      showConfirmButton: false,
      showCloseButton: false,
      allowOutsideClick: false,
      allowEscapeKey: false,
      allowEnterKey: false,
      focusConfirm: false,
      backdrop: 'rgba(17, 24, 39, .45)',
      customClass: { container: 'fb-swal-container', popup: 'fb-swal-popup' },
      didOpen: function () {
        setText('fb-load-title', opts.title || '正在处理');
        setText('fb-load-note', opts.note || '');
        tick();
        state.timer = setInterval(tick, 200);
        spawn();
        state.spawner = setInterval(spawn, 2400);
      },
      willClose: function () {
        clearInterval(state.timer);
        clearInterval(state.spawner);
        state.timer = null;
        state.spawner = null;
        state.open = false;
      }
    });
  }

  window.FBLoad = {
    show: function (opts) {
      opts = opts || {};
      state.opens += 1;
      if (state.open) {
        setText('fb-load-title', opts.title || '正在处理');
        setText('fb-load-note', opts.note || '');
        return;
      }
      clearTimeout(state.pending);
      state.pending = setTimeout(function () { open(opts); }, opts.delay || 0);
    },
    close: function () {
      state.opens = Math.max(0, state.opens - 1);
      if (state.opens > 0) return;
      clearTimeout(state.pending);
      state.pending = null;
      if (!state.open) return;
      if (typeof Swal !== 'undefined') Swal.close();
    }
  };
})();
"""

HEAD = f"<style>{_CSS}</style><script>{_LIB}</script><script>{_SCRIPT}</script>"

# A Gradio `js` hook must hand the inputs back: whatever it returns becomes the
# payload, so an implicit undefined would blank every field of the request.
CLOSE_JS = "(...args) => { if (window.FBLoad) window.FBLoad.close(); return args; }"


def payload(
    title: str,
    note: str = "",
    delay: int = 0,
    phases: list[tuple[int, str]] | None = None,
) -> dict:
    """The object the browser needs to render one wait."""
    return {
        "title": title,
        "note": note,
        "delay": delay,
        "phases": [list(item) for item in (phases or [])],
    }


def open_js(
    title: str,
    note: str = "",
    delay: int = 0,
    phases: list[tuple[int, str]] | None = None,
) -> str:
    """A Gradio `js` snippet that opens the dialog with these labels."""
    return "(...args) => { if (window.FBLoad) window.FBLoad.show(%s); return args; }" % json.dumps(
        payload(title, note, delay, phases), ensure_ascii=False
    )


def open_js_mode(fast: dict, enhanced: dict) -> str:
    """Open with different copy depending on the retrieval mode on screen."""
    table = json.dumps({"fast": fast, "enhanced": enhanced}, ensure_ascii=False)
    return (
        "(...args) => { var m = document.documentElement.getAttribute('data-fb-mode')"
        " === 'enhanced' ? 'enhanced' : 'fast';"
        " if (window.FBLoad) window.FBLoad.show(%s[m]); return args; }" % table
    )
