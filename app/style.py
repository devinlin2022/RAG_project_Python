"""Design tokens and CSS for the Gradio shell.

Class prefix `fb-`. Rules stay scoped to the Gradio container so the
framework's own components keep working.
"""

from __future__ import annotations

from app.loading import HEAD as LOADING_HEAD

HEAD = """
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
</style>
<script>
/* One job: tell the stylesheet which retrieval mode is selected, so the
   professional panels can be dimmed in the business view. Gradio renders
   after DOMContentLoaded, so we wait for the fieldset instead of assuming it.
   If this never runs, every panel stays available — a degraded look, never a
   broken app. */
(function () {
  var tries = 0;
  function current() {
    var box = document.querySelector('#fb-mode');
    if (!box) return null;
    var on = box.querySelector('label.selected, label[data-selected="true"]');
    if (!on) {
      var checked = box.querySelector('input:checked');
      if (checked && checked.value) return checked.value;
      return null;
    }
    var text = (on.textContent || '').trim();
    if (text.indexOf('增强') >= 0) return 'enhanced';
    if (text.indexOf('快速') >= 0) return 'fast';
    return null;
  }
  function sync() {
    var value = current();
    if (value) document.documentElement.setAttribute('data-fb-mode', value);
  }
  function attach(box) {
    ['change', 'click', 'input'].forEach(function (name) {
      box.addEventListener(name, function () { setTimeout(sync, 10); }, true);
    });
    new MutationObserver(sync).observe(box, {
      subtree: true, childList: true, attributes: true,
      attributeFilter: ['class', 'checked', 'aria-checked']
    });
    sync();
  }
  function boot() {
    var box = document.querySelector('#fb-mode');
    if (box) { attach(box); return; }
    if (tries++ < 60) setTimeout(boot, 250);   /* ~15 s of retries while Gradio boots */
  }
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', boot);
  } else {
    boot();
  }
})();
</script>
"""

CSS = """
/* ───────────────────────────── tokens ───────────────────────────── */
:root{
  --fb-bg:#f9fafb;
  --fb-surface:#ffffff;
  --fb-surface-2:#f9fafb;
  --fb-surface-3:#f3f4f6;
  --fb-border:#e5e7eb;
  --fb-border-2:#d1d5db;
  --fb-ink:#111827;
  --fb-ink-2:#4b5563;
  --fb-ink-3:#9ca3af;
  --fb-primary:#1a56db;
  --fb-primary-600:#1c64f2;
  --fb-primary-soft:#eff6ff;
  --fb-green:#057a55;
  --fb-green-soft:#def7ec;
  --fb-amber:#b45309;
  --fb-amber-soft:#fef3c7;
  --fb-red:#e02424;
  --fb-red-soft:#fde8e8;
  --fb-r:8px;
  --fb-r-lg:12px;
  --fb-shadow:0 1px 2px 0 rgb(16 24 40 / .05);
  --fb-shadow-md:0 4px 6px -1px rgb(16 24 40 / .08), 0 2px 4px -2px rgb(16 24 40 / .06);
  --fb-shadow-lg:0 10px 15px -3px rgb(16 24 40 / .1), 0 4px 6px -4px rgb(16 24 40 / .05);
  --fb-font:"Inter","PingFang SC","Microsoft YaHei",ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;
}
.dark{
  --fb-bg:#111827;
  --fb-surface:#1f2937;
  --fb-surface-2:#111827;
  --fb-surface-3:#374151;
  --fb-border:#374151;
  --fb-border-2:#4b5563;
  --fb-ink:#f9fafb;
  --fb-ink-2:#d1d5db;
  --fb-ink-3:#9ca3af;
  --fb-primary:#3f83f8;
  --fb-primary-soft:#1e3a8a;
  --fb-green-soft:#054f31;
  --fb-amber-soft:#4d2c00;
  --fb-red-soft:#4b1010;
  --fb-shadow:0 1px 2px 0 rgb(0 0 0 / .4);
  --fb-shadow-md:0 4px 6px -1px rgb(0 0 0 / .4);
  --fb-shadow-lg:0 10px 15px -3px rgb(0 0 0 / .5);
}

/* ─────────────────────── base / gradio reset ─────────────────────── */
body, .gradio-container{
  background:var(--fb-bg) !important;
  font-family:var(--fb-font) !important;
  color:var(--fb-ink) !important;
}
.gradio-container{ max-width:100% !important; padding:0 !important; }
.gradio-container .prose{ color:var(--fb-ink); font-size:13.5px; line-height:1.65; }
.gradio-container .prose h1,.gradio-container .prose h2,
.gradio-container .prose h3,.gradio-container .prose h4{
  font-weight:600; letter-spacing:-.011em; color:var(--fb-ink);
}
footer{ display:none !important; }
.gradio-container .block{ border:0 !important; background:transparent !important; padding:0 !important; }
.gradio-container .block.padded{ padding:0 !important; }

/* ─────────────────────────── app shell ─────────────────────────── */
#fb-shell{
  display:grid !important;
  grid-template-columns:268px minmax(0,1fr);
  gap:20px;
  align-items:start;
  padding:18px 20px 28px;
  max-width:1680px;
  margin:0 auto;
}
#fb-shell > .column{ min-width:0 !important; }
#fb-rail{
  position:sticky; top:14px;
  background:var(--fb-surface);
  border:1px solid var(--fb-border);
  border-radius:var(--fb-r-lg);
  box-shadow:var(--fb-shadow);
  padding:14px 12px 12px !important;
  gap:8px !important;
  min-width:0 !important;
}
#fb-main{ gap:14px !important; }

/* rail: brand */
.fb-brand{ display:flex; align-items:center; gap:10px; padding:2px 4px 12px; }
.fb-brand-mark{
  width:34px; height:34px; flex:0 0 34px; border-radius:9px;
  display:grid; place-items:center; color:#fff;
  background:linear-gradient(135deg,#1c64f2 0%,#1a56db 55%,#0e7490 100%);
  box-shadow:0 2px 6px rgb(26 86 219 / .35);
}
.fb-brand-mark svg{ width:18px; height:18px; }
.fb-brand-name{ font-size:14px; font-weight:600; color:var(--fb-ink); letter-spacing:-.01em; line-height:1.2; }
.fb-brand-sub{ font-size:11px; color:var(--fb-ink-3); margin-top:2px; letter-spacing:.01em; }

/* rail: section captions */
.fb-cap{
  font-size:10.5px; font-weight:600; letter-spacing:.08em; text-transform:uppercase;
  color:var(--fb-ink-3); padding:10px 6px 4px;
}
.fb-cap-row{ display:flex; align-items:center; justify-content:space-between; gap:6px; }

/* rail: mode switch (Gradio Radio → segmented control) */
#fb-mode{ margin:2px 0 2px !important; }
#fb-mode .wrap, #fb-mode > div{
  display:grid !important;
  grid-template-columns:1fr 1fr;
  gap:3px;
  background:var(--fb-surface-3);
  border-radius:9px;
  padding:3px;
  border:0 !important;
}
#fb-mode label{
  margin:0 !important;
  padding:6px 8px !important;
  border:0 !important;
  border-radius:7px !important;
  background:transparent !important;
  color:var(--fb-ink-2) !important;
  font-size:12.5px !important;
  font-weight:500 !important;
  text-align:center;
  cursor:pointer;
  display:flex; align-items:center; justify-content:center;
  transition:background .15s ease, color .15s ease, box-shadow .15s ease;
}
#fb-mode label:hover{ color:var(--fb-ink) !important; }
#fb-mode label.selected,
#fb-mode label:has(input:checked){
  background:var(--fb-surface) !important;
  color:var(--fb-primary) !important;
  font-weight:600 !important;
  box-shadow:0 1px 2px rgb(16 24 40 / .12);
}
#fb-mode input{ display:none !important; }

/* rail: nav buttons */
#fb-nav{ gap:2px !important; }
#fb-nav button{
  width:100% !important;
  min-height:34px !important;
  justify-content:flex-start !important;
  display:flex !important; align-items:center !important;
  gap:0 !important;
  padding:7px 9px !important;
  border:0 !important; border-radius:7px !important;
  background:transparent !important; box-shadow:none !important;
  color:var(--fb-ink-2) !important;
  font-size:13px !important; font-weight:500 !important;
  text-align:left !important;
  transition:background .12s ease, color .12s ease;
}
#fb-nav button:hover{ background:var(--fb-surface-3) !important; color:var(--fb-ink) !important; }
#fb-nav button.primary{
  background:var(--fb-primary-soft) !important;
  color:var(--fb-primary) !important;
  font-weight:600 !important;
}
#fb-nav button::before{
  content:""; width:16px; height:16px; flex:0 0 16px; margin-right:9px;
  background:currentColor;
  -webkit-mask-repeat:no-repeat; mask-repeat:no-repeat;
  -webkit-mask-position:center; mask-position:center;
  -webkit-mask-size:16px 16px; mask-size:16px 16px;
  opacity:.85;
}
#nav-chat button::before, #nav-chat::before{ -webkit-mask-image:var(--fb-ico-chat); mask-image:var(--fb-ico-chat); }
#nav-docs button::before, #nav-docs::before{ -webkit-mask-image:var(--fb-ico-doc); mask-image:var(--fb-ico-doc); }
#nav-chunks button::before, #nav-chunks::before{ -webkit-mask-image:var(--fb-ico-grid); mask-image:var(--fb-ico-grid); }
#nav-pages button::before, #nav-pages::before{ -webkit-mask-image:var(--fb-ico-table); mask-image:var(--fb-ico-table); }
#nav-text button::before, #nav-text::before{ -webkit-mask-image:var(--fb-ico-text); mask-image:var(--fb-ico-text); }
#nav-metrics button::before, #nav-metrics::before{ -webkit-mask-image:var(--fb-ico-chart); mask-image:var(--fb-ico-chart); }
#nav-system button::before, #nav-system::before{ -webkit-mask-image:var(--fb-ico-gear); mask-image:var(--fb-ico-gear); }

/* static element ids land on the button itself in some versions */
button#nav-chat, button#nav-docs, button#nav-chunks, button#nav-pages,
button#nav-text, button#nav-metrics, button#nav-system{
  width:100% !important; justify-content:flex-start !important;
  min-height:34px !important; padding:7px 9px !important;
  border:0 !important; border-radius:7px !important;
  background:transparent !important; box-shadow:none !important;
  color:var(--fb-ink-2) !important; font-size:13px !important;
  font-weight:500 !important; text-align:left !important;
}
button#nav-chat.primary, button#nav-docs.primary, button#nav-chunks.primary,
button#nav-pages.primary, button#nav-text.primary, button#nav-metrics.primary,
button#nav-system.primary{
  background:var(--fb-primary-soft) !important; color:var(--fb-primary) !important;
  font-weight:600 !important;
}

/* lock state: business view dims the professional entries */
.fb-lock{
  display:none; margin-left:auto; width:12px; height:12px; flex:0 0 12px;
  background:currentColor; opacity:.6;
  -webkit-mask:var(--fb-ico-lock) center/12px no-repeat;
  mask:var(--fb-ico-lock) center/12px no-repeat;
}
html[data-fb-mode="fast"] #fb-nav button.fb-pro{ opacity:.38; pointer-events:none; filter:saturate(.2); }
html[data-fb-mode="fast"] #fb-rail .fb-pro-hint{ display:flex; }
.fb-pro-hint{
  display:none; align-items:flex-start; gap:6px;
  margin:8px 2px 0; padding:8px 9px;
  background:var(--fb-amber-soft); border-radius:7px;
  font-size:11px; line-height:1.45; color:var(--fb-amber);
}
.fb-pro-hint b{ font-weight:600; }

/* rail: model status footer */
.fb-status{ display:flex; flex-direction:column; gap:5px; padding:10px 6px 2px;
  border-top:1px solid var(--fb-border); margin-top:6px; }
.fb-status-row{ display:flex; align-items:center; gap:6px; font-size:11px; color:var(--fb-ink-3); }
.fb-dot{ width:6px; height:6px; border-radius:999px; flex:0 0 6px; background:var(--fb-green); }
.fb-dot.off{ background:var(--fb-ink-3); }
.fb-dot.warn{ background:#f59e0b; }

/* ─────────────────────────── primitives ─────────────────────────── */
.fb-card{
  background:var(--fb-surface);
  border:1px solid var(--fb-border);
  border-radius:var(--fb-r-lg);
  box-shadow:var(--fb-shadow);
  overflow:hidden;
}
.fb-card-head{
  display:flex; align-items:center; justify-content:space-between; gap:10px;
  padding:12px 16px; border-bottom:1px solid var(--fb-border);
  background:var(--fb-surface);
}
.fb-card-title{ font-size:13.5px; font-weight:600; color:var(--fb-ink); letter-spacing:-.01em; }
.fb-card-sub{ font-size:11.5px; color:var(--fb-ink-3); margin-top:2px; }
.fb-card-body{ padding:14px 16px; }
.fb-card-body.tight{ padding:10px 12px; }

.fb-page-head{ padding:0 0 4px; }
.fb-page-title{ font-size:19px; font-weight:600; letter-spacing:-.02em; color:var(--fb-ink); }
.fb-page-desc{ font-size:12.5px; color:var(--fb-ink-2); margin-top:4px; line-height:1.6; max-width:70ch; }

.fb-badge{
  display:inline-flex; align-items:center; gap:4px;
  padding:1px 7px; border-radius:999px;
  font-size:11px; font-weight:500; line-height:18px;
  background:var(--fb-surface-3); color:var(--fb-ink-2);
}
.fb-badge.blue{ background:var(--fb-primary-soft); color:var(--fb-primary); }
.fb-badge.green{ background:var(--fb-green-soft); color:var(--fb-green); }
.fb-badge.amber{ background:var(--fb-amber-soft); color:var(--fb-amber); }
.fb-badge.red{ background:var(--fb-red-soft); color:var(--fb-red); }
.fb-badge.outline{ background:transparent; border:1px solid var(--fb-border-2); }

.fb-kv{ display:grid; grid-template-columns:repeat(auto-fit,minmax(96px,1fr)); gap:8px; }
.fb-kv-item{ background:var(--fb-surface-2); border:1px solid var(--fb-border);
  border-radius:var(--fb-r); padding:8px 10px; }
.fb-kv-num{ font-size:15px; font-weight:600; color:var(--fb-ink); letter-spacing:-.01em; }
.fb-kv-lab{ font-size:10.5px; color:var(--fb-ink-3); margin-top:1px; letter-spacing:.02em; }

.fb-empty{ text-align:center; padding:26px 16px; color:var(--fb-ink-3); font-size:12.5px; }
.fb-empty svg{ width:26px; height:26px; margin-bottom:8px; opacity:.5; }

/* ───────────────────── gradio controls restyled ───────────────────── */
.gradio-container label > span,
.gradio-container .block > label > span{
  font-size:11.5px !important; font-weight:500 !important;
  color:var(--fb-ink-2) !important; letter-spacing:.01em;
}
.gradio-container input[type="text"],
.gradio-container input[type="password"],
.gradio-container input[type="number"],
.gradio-container textarea,
.gradio-container select{
  border:1px solid var(--fb-border-2) !important;
  border-radius:var(--fb-r) !important;
  background:var(--fb-surface-2) !important;
  color:var(--fb-ink) !important;
  font-size:13px !important;
  font-family:var(--fb-font) !important;
  box-shadow:none !important;
  padding:8px 10px !important;
  transition:border-color .12s ease, box-shadow .12s ease, background .12s ease;
}
.gradio-container input:focus, .gradio-container textarea:focus, .gradio-container select:focus{
  border-color:var(--fb-primary) !important;
  background:var(--fb-surface) !important;
  box-shadow:0 0 0 3px rgb(26 86 219 / .12) !important;
  outline:none !important;
}
.gradio-container .wrap-inner, .gradio-container .secondary-wrap{
  border-radius:var(--fb-r) !important;
  border-color:var(--fb-border-2) !important;
  background:var(--fb-surface-2) !important;
}
.gradio-container button.primary{
  background:var(--fb-primary) !important; border-color:var(--fb-primary) !important;
  color:#fff !important; font-weight:600 !important; font-size:13px !important;
  border-radius:var(--fb-r) !important; box-shadow:var(--fb-shadow) !important;
}
.gradio-container button.primary:hover{ background:var(--fb-primary-600) !important; }
.gradio-container button.secondary{
  background:var(--fb-surface) !important; border:1px solid var(--fb-border-2) !important;
  color:var(--fb-ink) !important; font-weight:500 !important; font-size:13px !important;
  border-radius:var(--fb-r) !important; box-shadow:var(--fb-shadow) !important;
}
.gradio-container button.stop{
  background:var(--fb-surface) !important; border:1px solid #f3c1c1 !important;
  color:var(--fb-red) !important; border-radius:var(--fb-r) !important;
  font-weight:500 !important; font-size:13px !important;
}
.gradio-container .slider_input, .gradio-container .head{ font-size:12px !important; color:var(--fb-ink-2) !important; }

/* dataframes / tables */
.gradio-container table{ font-size:12px !important; font-family:var(--fb-font) !important; }
.gradio-container thead th{
  background:var(--fb-surface-2) !important; color:var(--fb-ink-2) !important;
  font-weight:600 !important; font-size:11px !important; letter-spacing:.02em;
  border-bottom:1px solid var(--fb-border) !important; padding:7px 10px !important;
}
.gradio-container tbody td{
  border-bottom:1px solid var(--fb-border) !important;
  color:var(--fb-ink) !important; padding:6px 10px !important;
}
.gradio-container tbody tr:hover td{ background:var(--fb-surface-2) !important; }

/* accordions */
.gradio-container .label-wrap{ font-size:13px !important; font-weight:600 !important; color:var(--fb-ink) !important; }
.gradio-container details{ border:1px solid var(--fb-border) !important; border-radius:var(--fb-r) !important;
  background:var(--fb-surface) !important; padding:2px 12px !important; }

/* ─────────────────────────── chat room ─────────────────────────── */
#fb-chatpage{
  display:grid !important;
  grid-template-columns:minmax(0,1fr) 340px;
  gap:16px; align-items:start;
}
#fb-chatpage > .column{ min-width:0 !important; }

#fb-chatcard{
  background:var(--fb-surface); border:1px solid var(--fb-border);
  border-radius:var(--fb-r-lg); box-shadow:var(--fb-shadow);
  overflow:hidden; display:flex; flex-direction:column;
}
.fb-chat-head{
  display:flex; align-items:center; gap:12px;
  padding:12px 16px; border-bottom:1px solid var(--fb-border);
  background:var(--fb-surface);
}
.fb-chat-avatar{
  width:36px; height:36px; flex:0 0 36px; border-radius:999px;
  display:grid; place-items:center; color:#fff;
  background:linear-gradient(135deg,#1c64f2,#0e7490);
  box-shadow:0 1px 4px rgb(26 86 219 / .3);
}
.fb-chat-avatar svg{ width:18px; height:18px; }
.fb-chat-name{ font-size:13.5px; font-weight:600; color:var(--fb-ink); display:flex; align-items:center; gap:6px; }
.fb-chat-meta{ font-size:11.5px; color:var(--fb-ink-3); margin-top:2px; }
.fb-live{ display:inline-flex; align-items:center; gap:4px; font-size:10.5px; color:var(--fb-green); }
.fb-live::before{ content:""; width:5px; height:5px; border-radius:999px; background:var(--fb-green); }

/* the chatbot itself — one turn is .message-row > .flex-wrap > .message */
#fb-chat{ border:0 !important; background:transparent !important; }
#fb-chat .bubble-wrap{ background:transparent !important; padding:16px 16px 4px !important; }
#fb-chat .message-wrap{ gap:12px !important; padding:0 !important; }
#fb-chat .icon-button-wrapper,
#fb-chat .message-buttons{ display:none !important; }   /* the composer has its own clear */
#fb-chat .message-row{ display:flex !important; width:100% !important; padding:0 !important; }
#fb-chat .message-row.user-row{ justify-content:flex-end !important; }
#fb-chat .message-row.bot-row{ justify-content:flex-start !important; }
#fb-chat .message-row .flex-wrap{ display:flex !important; width:100% !important; }
#fb-chat .message-row.user-row .flex-wrap{ justify-content:flex-end !important; }
/* avatar on the assistant side, the way the Flowbite chat room shows it */
#fb-chat .message-row.bot-row::before{
  content:"文"; display:grid; place-items:center;
  width:30px; height:30px; flex:0 0 30px; border-radius:999px;
  margin:2px 10px 0 0;
  background:linear-gradient(135deg,#1c64f2,#0e7490);
  color:#fff; font-size:12px; font-weight:600; line-height:1;
}
#fb-chat .message{
  width:auto !important;
  max-width:min(88%,620px) !important;
  padding:10px 14px !important;
  border-radius:12px !important;
  border-top-left-radius:4px !important;
  background:var(--fb-surface) !important;
  border:1px solid var(--fb-border) !important;
  box-shadow:var(--fb-shadow) !important;
  color:var(--fb-ink) !important;
}
#fb-chat .message.user{
  background:var(--fb-primary) !important;
  border-color:var(--fb-primary) !important;
  border-top-left-radius:12px !important;
  border-top-right-radius:4px !important;
  box-shadow:0 2px 8px rgb(26 86 219 / .22) !important;
  color:#fff !important;
}
#fb-chat .message.user .md, #fb-chat .message.user .md *{ color:#fff !important; }
#fb-chat .message-content{ background:transparent !important; padding:0 !important; }
#fb-chat .message .md{ font-size:13.5px !important; line-height:1.68 !important; }
#fb-chat .message p{ margin:0 0 6px !important; }
#fb-chat .message p:last-child{ margin-bottom:0 !important; }
#fb-chat .message h1,#fb-chat .message h2,#fb-chat .message h3{
  font-size:13.5px !important; font-weight:600 !important; margin:8px 0 4px !important;
}
#fb-chat .message ul,#fb-chat .message ol{ margin:4px 0 6px !important; padding-left:18px !important; }
#fb-chat .message li{ margin:2px 0 !important; }
#fb-chat .message table{ width:100% !important; border-collapse:collapse !important; margin:6px 0 !important; }
#fb-chat .message th,
#fb-chat .message td{
  border:1px solid var(--fb-border) !important; padding:4px 8px !important;
  font-size:12px !important;
}
#fb-chat .message.user th, #fb-chat .message.user td{ border-color:rgb(255 255 255 / .35) !important; }
#fb-chat .message code{
  background:var(--fb-surface-3) !important; border-radius:4px !important;
  padding:1px 4px !important; font-size:12px !important;
}
#fb-chat .message.user code{ background:rgb(255 255 255 / .2) !important; }
#fb-chat .message a{ color:var(--fb-primary) !important; text-decoration:underline !important; }
#fb-chat .message.user a{ color:#fff !important; }
#fb-chat .panel, #fb-chat .bubble-border{ display:none !important; }

/* home upload row: keep it on one line */
#fb-chat-upload{
  display:flex !important; flex-wrap:nowrap !important; align-items:stretch !important;
  gap:8px !important; margin:0 14px 6px !important;
}
#fb-chat-upload > *{ min-width:0 !important; }
#fb-chat-drop{ flex:1 1 0 !important; min-width:0 !important; }
#fb-chat-drop > .block{ width:100% !important; }
/* Gradio file picker is 119px tall by default; squeeze it to one line */
#fb-chat-drop .upload-container{
  padding:6px 10px !important; min-height:0 !important; gap:8px !important;
  border:1px dashed var(--fb-border-2) !important; border-radius:12px !important;
  background:var(--fb-surface-2) !important;
}
#fb-chat-drop .upload-container .wrap{ height:auto !important; min-height:0 !important; }
#fb-chat-drop .icon-wrap svg{ width:18px !important; height:18px !important; }
#fb-chat-drop .or{ display:none !important; }
#fb-chat-drop .wrap.default.full{ min-height:0 !important; }
#fb-chat-drop label{ font-size:12px !important; color:var(--fb-ink-3) !important; margin-bottom:2px !important; }
#fb-chat-drop .file-preview{ max-height:56px; overflow:auto; }
#fb-chat-upload-btn{ flex:0 0 116px !important; height:auto !important; min-height:56px !important; }
#fb-chat-note{ margin:0 14px !important; }
#fb-chat-note .prose{ font-size:12px !important; }
#fb-chat-note .prose ul{ margin:2px 0 6px !important; padding-left:16px; }

/* document table: smaller type once the accordion narrows the column */
#fb-doc-table{ --cell-padding: 6px 8px; }
#fb-doc-table table, #fb-doc-table td, #fb-doc-table th{ font-size:12px !important; }
#fb-doc-table th{ white-space:nowrap !important; }

/* composer */
#fb-composer{
  display:flex !important; align-items:flex-end !important; gap:8px !important;
  margin:8px 14px 14px !important;
  padding:6px !important;
  border:1px solid var(--fb-border-2) !important;
  border-radius:14px !important;
  background:var(--fb-surface-2) !important;
  box-shadow:0 1px 2px rgb(16 24 40 / .04);
}
#fb-composer:focus-within{
  border-color:var(--fb-primary) !important;
  box-shadow:0 0 0 3px rgb(26 86 219 / .12) !important;
  background:var(--fb-surface) !important;
}
#fb-composer .block{ border:0 !important; background:transparent !important; padding:0 !important; }
#fb-composer textarea{
  border:0 !important; background:transparent !important; box-shadow:none !important;
  padding:9px 10px !important; font-size:13.5px !important; resize:none !important;
  min-height:40px !important;
}
#fb-composer textarea:focus{ box-shadow:none !important; background:transparent !important; }
#fb-composer button{
  flex:0 0 44px !important; height:40px !important;
  width:44px !important; min-width:44px !important; max-width:44px !important;
  padding:0 !important; border-radius:11px !important;
  display:grid !important; place-items:center;
  font-size:0 !important; line-height:0 !important;
}
#fb-send{
  background:var(--fb-primary) !important; border:0 !important; color:#fff !important;
  box-shadow:0 2px 6px rgb(26 86 219 / .3) !important;
}
#fb-send::after{
  content:""; width:17px; height:17px; background:#fff;
  -webkit-mask:var(--fb-ico-send) center/17px no-repeat;
  mask:var(--fb-ico-send) center/17px no-repeat;
}
#fb-clear{ background:transparent !important; border:0 !important; color:var(--fb-ink-3) !important; }
#fb-clear::after{
  content:""; width:16px; height:16px; background:currentColor;
  -webkit-mask:var(--fb-ico-trash) center/16px no-repeat;
  mask:var(--fb-ico-trash) center/16px no-repeat;
}
.fb-composer-hint{
  padding:0 18px 12px; font-size:11px; color:var(--fb-ink-3);
  display:flex; align-items:center; gap:6px;
}
.fb-composer-hint kbd{
  background:var(--fb-surface-3); border:1px solid var(--fb-border);
  border-bottom-width:2px; border-radius:4px; padding:0 4px;
  font-size:10px; font-family:var(--fb-font); color:var(--fb-ink-2);
}

/* inspector column */
#fb-side{ gap:12px !important; }

/* pipeline timeline */
.fb-trace{ padding:4px 0 2px; }
.fb-trace-top{
  display:flex; align-items:center; justify-content:space-between;
  padding:0 0 8px; font-size:11px; color:var(--fb-ink-3);
}
.fb-progress{ height:3px; border-radius:999px; background:var(--fb-surface-3); overflow:hidden; margin-bottom:10px; }
.fb-progress > i{ display:block; height:100%; border-radius:999px; background:var(--fb-green); transition:width .3s ease; }
.fb-step{ display:flex; gap:10px; padding:0 0 12px; position:relative; }
.fb-step:last-child{ padding-bottom:2px; }
.fb-step::before{
  content:""; position:absolute; left:8px; top:20px; bottom:0; width:1px;
  background:var(--fb-border);
}
.fb-step:last-child::before{ display:none; }
.fb-step-dot{
  flex:0 0 17px; width:17px; height:17px; border-radius:999px;
  display:grid; place-items:center; font-size:9.5px; font-weight:700;
  background:var(--fb-surface); border:1.5px solid var(--fb-border-2);
  color:var(--fb-ink-3); z-index:1; line-height:1;
}
.fb-step.done .fb-step-dot{ background:var(--fb-green-soft); border-color:var(--fb-green); color:var(--fb-green); }
.fb-step.running .fb-step-dot{ border-color:var(--fb-primary); color:var(--fb-primary); }
.fb-step.running .fb-step-dot i{
  display:block; width:9px; height:9px; border-radius:999px;
  border:1.5px solid currentColor; border-top-color:transparent;
  animation:fb-spin .9s linear infinite;
}
.fb-step.error .fb-step-dot{ background:var(--fb-red-soft); border-color:var(--fb-red); color:var(--fb-red); }
.fb-step.skip .fb-step-dot{ background:var(--fb-amber-soft); border-color:#f59e0b; color:var(--fb-amber); }
.fb-step-body{ min-width:0; flex:1 1 auto; }
.fb-step-name{ font-size:12.5px; font-weight:600; color:var(--fb-ink); display:flex; align-items:baseline; gap:6px; }
.fb-step.done .fb-step-name{ color:var(--fb-ink); }
.fb-step-name em{ font-style:normal; font-weight:400; font-size:10.5px; color:var(--fb-ink-3); margin-left:auto; }
.fb-step-detail{ font-size:11.5px; color:var(--fb-ink-2); line-height:1.5; margin-top:2px; word-break:break-word; }
@keyframes fb-spin{ to{ transform:rotate(360deg); } }

/* chunk map: page strip + bars */
.fb-strip{ display:flex; gap:2px; margin:2px 0 6px; }
.fb-strip i{
  flex:1 1 0; min-width:3px; height:18px; border-radius:2px;
  background:var(--fb-surface-3); display:block;
}
.fb-strip i.cited{ box-shadow:0 0 0 1.5px var(--fb-red); }
.fb-strip-note{ font-size:10.5px; color:var(--fb-ink-3); margin-bottom:10px; display:flex; gap:10px; flex-wrap:wrap; }
.fb-map{ max-height:360px; overflow:auto; margin:0 -4px; padding:0 4px; }
.fb-map-row{
  display:grid; grid-template-columns:40px 52px 34px 1fr 62px;
  align-items:center; gap:8px; padding:4px 6px; border-radius:6px;
  font-size:11.5px; color:var(--fb-ink-2);
}
.fb-map-row:hover{ background:var(--fb-surface-2); }
.fb-map-row.cited{ background:var(--fb-red-soft); }
.fb-map-idx{ color:var(--fb-ink-3); font-variant-numeric:tabular-nums; }
.fb-map-page{ color:var(--fb-ink-2); font-variant-numeric:tabular-nums; }
.fb-map-kind{ color:var(--fb-ink-3); font-size:10.5px; }
.fb-track{ height:7px; border-radius:999px; background:var(--fb-surface-3); overflow:hidden; }
.fb-track > i{ display:block; height:100%; border-radius:999px; }
.fb-map-chars{ text-align:right; color:var(--fb-ink-3); font-variant-numeric:tabular-nums; }
.fb-map-preview{
  grid-column:1 / -1; font-size:11px; color:var(--fb-ink-3);
  white-space:nowrap; overflow:hidden; text-overflow:ellipsis;
  margin:-2px 0 4px 6px;
}
.fb-cite{
  display:inline-flex; align-items:center; justify-content:center;
  min-width:16px; height:16px; padding:0 4px; margin-left:5px;
  border-radius:999px; background:var(--fb-red); color:#fff;
  font-size:9.5px; font-weight:600; vertical-align:1px;
}
.fb-legend{ display:flex; gap:12px; flex-wrap:wrap; font-size:10.5px; color:var(--fb-ink-3); margin-top:8px; }
.fb-legend span{ display:inline-flex; align-items:center; gap:4px; }
.fb-legend b{ width:8px; height:8px; border-radius:2px; display:inline-block; }

/* stepper */
.fb-stepper{ display:flex; gap:6px; overflow-x:auto; padding:0 0 2px; }
.fb-step-chip{
  flex:1 1 0; min-width:104px;
  border:1px solid var(--fb-border); border-radius:8px;
  padding:7px 9px; background:var(--fb-surface);
}
.fb-step-chip.done{ background:var(--fb-green-soft); border-color:#a7e0c0; }
.fb-step-chip b{ display:block; font-size:11px; font-weight:600; color:var(--fb-ink); letter-spacing:-.005em; }
.fb-step-chip.done b{ color:var(--fb-green); }
.fb-step-chip span{ display:block; font-size:10px; color:var(--fb-ink-3); margin-top:1px;
  white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }

/* panel grid */
.fb-split{ display:grid; grid-template-columns:minmax(0,1fr) 340px; gap:16px; align-items:start; }
.fb-grid-2{ display:grid; grid-template-columns:repeat(auto-fit,minmax(280px,1fr)); gap:14px; align-items:start; }
.fb-stack{ display:flex; flex-direction:column; gap:12px; }
.fb-row{ display:flex; gap:8px; align-items:flex-end; flex-wrap:wrap; }
.fb-row > *{ min-width:0; }
.fb-mt{ margin-top:10px; }


/* ───────────────────── panel wiring (app shell) ───────────────────── */
#fb-main{ gap:12px !important; }
#fb-flowbar{ margin:0 !important; }
#fb-flowbar .fb-stepper{ padding:0; }
#panel-chat{ gap:0 !important; }

/* workbench pages sit on one white sheet instead of a border per control */
#panel-docs, #panel-chunks, #panel-pages, #panel-text, #panel-metrics, #panel-system{
  background:var(--fb-surface);
  border:1px solid var(--fb-border);
  border-radius:var(--fb-r-lg);
  box-shadow:var(--fb-shadow);
  padding:18px !important;
  gap:14px !important;
}

#fb-chat-headrow{
  display:grid !important;
  grid-template-columns:minmax(0,1fr) auto;
  gap:10px 14px; align-items:center;
  padding:12px 14px 10px;
  border-bottom:1px solid var(--fb-border);
  background:var(--fb-surface);
}
/* Gradio gives every block `min-width: min(160px, 100%)`, which collapses a
   grid track against itself. Zero it here and let the tracks decide. */
#fb-chat-headrow > *{ min-width:0 !important; width:auto !important; }
#fb-chat-headrow > div:nth-child(2){ grid-column:1 / -1; grid-row:2; }
#fb-chat-headrow > div:nth-child(3){ grid-column:2; grid-row:1; justify-self:end; }

/* inspector accordions read as cards, not as raw disclosure rows */
/* `.gradio-container .block` zeroes border/background with !important, so
   these three need the same weight to win. */
#fb-main div:has(> button.label-wrap){
  background:var(--fb-surface) !important;
  border:1px solid var(--fb-border) !important;
  border-radius:var(--fb-r-lg) !important;
  box-shadow:var(--fb-shadow) !important;
  padding:0 14px 6px !important;
}
#fb-main button.label-wrap{
  width:100% !important;
  display:flex !important; align-items:center !important; justify-content:space-between !important;
  padding:12px 0 !important; margin:0 !important;
  background:transparent !important; border:0 !important; box-shadow:none !important;
  font-size:12.5px !important; font-weight:600 !important; color:var(--fb-ink) !important;
}
#fb-main button.label-wrap .icon{ font-size:9px !important; color:var(--fb-ink-3) !important; }
#fb-main [data-testid="accordion-content"] .block{ padding:0 0 8px !important; }
#fb-chat-headrow .fb-chat-head{ padding:0 !important; border:0 !important; background:transparent !important; }
#fb-chat-headrow .block{ border:0 !important; background:transparent !important; padding:0 !important; }
#fb-scope{ min-width:0 !important; }
#fb-scope .wrap-inner, #fb-scope .secondary-wrap{
  min-height:34px !important; font-size:12px !important;
}
.fb-badge-note{ font-size:10.5px; color:var(--fb-ink-3); margin-top:3px; text-align:right; }
.fb-hint-icon{ display:inline-flex; margin-top:1px; opacity:.85; }
.fb-hint-icon svg{ width:13px; height:13px; }

/* the answer's own numbers: mode, timings, tokens */
#fb-answer-meta{ padding:6px 16px 0 !important; }
#fb-answer-meta .prose{ font-size:11.5px !important; color:var(--fb-ink-3) !important; }
#fb-answer-meta ul, #fb-answer-meta .prose ul{
  list-style:none !important; display:flex !important; flex-wrap:wrap !important;
  gap:2px 16px !important; margin:0 !important; padding:0 !important;
}
#fb-answer-meta li, #fb-answer-meta .prose li{ margin:0 !important; line-height:1.6; }
#fb-answer-meta .prose p{ margin:0 !important; }
#fb-answer-meta .prose strong{ color:var(--fb-ink-2) !important; font-weight:600; }

/* workbench grids */
#fb-docs-grid{ display:grid !important; grid-template-columns:minmax(0,5fr) minmax(0,7fr); gap:16px; align-items:start; }
/* Gradio merges sibling controls without `scale` into one form container, so
   these bars are flex rows with explicit basis rather than fixed column counts. */
#fb-toolbar{ display:flex !important; flex-wrap:wrap; gap:10px; align-items:end; }
#fb-toolbar > *{ flex:1 1 150px; min-width:0 !important; }
#fb-toolbar > button{ flex:0 0 auto !important; }
#fb-parse-grid{ display:grid !important; grid-template-columns:1fr 1fr; gap:14px; align-items:start; }
#fb-chunk-grid{ display:flex !important; gap:14px; align-items:flex-start; }
#fb-chunk-grid > *:first-child{ flex:0 0 240px; }
#fb-chunk-grid > *:last-child{ flex:1 1 0; }
#fb-text-grid{ display:flex !important; gap:14px; align-items:flex-start; }
#fb-text-grid > *:first-child{ flex:0 0 140px; }
#fb-text-grid > *:last-child{ flex:1 1 0; }
#fb-metrics-bar{ display:grid !important; grid-template-columns:auto minmax(0,1fr); gap:12px; align-items:end; }
#fb-keybar{ display:grid !important; grid-template-columns:minmax(0,1fr) auto auto; gap:10px; align-items:end; }
#fb-docs-grid .block, #fb-toolbar .block, #fb-chunk-grid .block,
#fb-text-grid .block, #fb-metrics-bar .block, #fb-keybar .block{
  border:0 !important; background:transparent !important; padding:0 !important;
}
/* Gradio ships `min-width: min(160px, 100%)` on every block; inside a grid
   that resolves against the track being measured and collapses it. */
#fb-docs-grid > *, #fb-parse-grid > *, #fb-toolbar > *, #fb-chunk-grid > *,
#fb-text-grid > *, #fb-metrics-bar > *, #fb-keybar > *{ min-width:0 !important; }
#fb-toolbar button, #fb-metrics-bar button, #fb-keybar button{
  width:auto !important; min-width:92px !important;
}

/* component labels should not paint a grey pill above the control */
.gradio-container label > span{ background:transparent !important; padding-left:0 !important; }
/* Gradio's `form` wrapper tints grouped controls; that band reads as a
   half-finished section header, so drop it and let the inputs carry the tone. */
.gradio-container .form, .gradio-container .form > .block,
.gradio-container .block.padded, .gradio-container .block.hide-container{
  background:transparent !important; border:0 !important; box-shadow:none !important;
}
.gradio-container .block > label{ background:transparent !important; }
#panel-docs .gradio-file, #panel-docs .file-preview{ border-radius:var(--fb-r) !important; }
#panel-docs .block.accent{ border:0 !important; }

@media (max-width:1240px){
  #fb-docs-grid, #fb-parse-grid, #fb-metrics-bar{ grid-template-columns:minmax(0,1fr) !important; }
}

@media (max-width:1240px){
  #fb-chatpage{ grid-template-columns:minmax(0,1fr); }
  .fb-split{ grid-template-columns:minmax(0,1fr); }
}
@media (max-width:980px){
  #fb-shell{ grid-template-columns:minmax(0,1fr); padding:12px; }
  #fb-rail{ position:static; }
}
"""


# ─────────────────────────── icons ───────────────────────────
# Masked SVGs keep the nav icons monochrome and theme-aware: the colour comes
# from `background: currentColor`, so an active item tints its own icon.
_ICON_PATHS = {
    "chat": "M20.5 11.5a8 8 0 0 1-8.5 8 9 9 0 0 1-4-1L4 20l1.5-3.8A8 8 0 0 1 12 3.5a8 8 0 0 1 8.5 8Z",
    "doc": "M13.5 3.5H7A1.5 1.5 0 0 0 5.5 5v14A1.5 1.5 0 0 0 7 20.5h10a1.5 1.5 0 0 0 1.5-1.5V8.5l-5-5Zm0 0V8.5h5M9 13h6M9 16.5h4",
    "grid": "M4 4h6v6H4zM14 4h6v6h-6zM4 14h6v6H4zM14 14h6v6h-6z",
    "table": "M4 5.5h16v13H4zM4 10h16M10 10v8.5",
    "text": "M5 6.5h14M5 11h14M5 15.5h9",
    "chart": "M4.5 19V10M10 19V5M15.5 19v-6M21 19H3",
    "gear": "M12 15.2a3.2 3.2 0 1 0 0-6.4 3.2 3.2 0 0 0 0 6.4ZM4 12h1.9M18.1 12H20M12 4v1.9M12 18.1V20M6.4 6.4l1.3 1.3M16.3 16.3l1.3 1.3M17.6 6.4l-1.3 1.3M7.7 16.3l-1.3 1.3",
    "lock": "M7.5 10.5V8a4.5 4.5 0 0 1 9 0v2.5M5.5 10.5h13v9h-13z",
    "send": "M4 12l16-7-7 16-2.2-6.8L4 12Z",
    "trash": "M4.5 7h15M9.5 7V4.5h5V7M6.5 7l1 13h9l1-13",
}


def _icon_vars() -> str:
    from urllib.parse import quote

    lines = [":root{"]
    for name, path in _ICON_PATHS.items():
        svg = (
            "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none'"
            " stroke='black' stroke-width='1.9' stroke-linecap='round'"
            f" stroke-linejoin='round'><path d='{path}'/></svg>"
        )
        lines.append(f"  --fb-ico-{name}:url(\"data:image/svg+xml;utf8,{quote(svg)}\");")
    lines.append("}")
    return "\n".join(lines)


CSS = CSS + "\n" + _icon_vars()
HEAD = HEAD + LOADING_HEAD
