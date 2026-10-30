"""The look is code, so pin the parts that actually carry the design.

These are not pixel tests. They guard three decisions that are easy to undo by
accident: the token palette, the mode gate, and staying off a CDN.
"""

from app.style import CSS, HEAD


def test_palette_is_flowbite_not_default_gradio():
    for token in ("--fb-primary:#1a56db", "--fb-bg:#f9fafb", "--fb-border:#e5e7eb"):
        assert token in CSS
    assert "--fb-r-lg:12px" in CSS


def test_pro_panels_are_gated_by_mode():
    assert 'html[data-fb-mode="fast"] #fb-nav button.fb-pro' in CSS
    gate = CSS.split('html[data-fb-mode="fast"] #fb-nav button.fb-pro')[1].split("}")[0]
    assert "pointer-events:none" in gate
    assert "opacity" in gate


def test_layout_has_no_tailwind_dependency():
    assert "cdn.tailwindcss.com" not in CSS
    assert "tailwind" not in CSS.lower()
    assert "data:image/svg+xml" in CSS   # icons travel with the stylesheet


def test_head_waits_for_gradio_instead_of_domcontentloaded():
    assert "#fb-mode" in HEAD
    assert "MutationObserver" in HEAD
    assert "setTimeout(boot" in HEAD      # retries while the app renders


def test_chat_bubble_rules_target_the_real_dom():
    # Gradio renders .message-row > .message; styling .bubble alone repaints the
    # copy-button group as well, which is how the first attempt went wrong.
    assert "#fb-chat .message.user" in CSS
    assert "#fb-chat .message-row.bot-row::before" in CSS
    assert "#fb-chat .message-buttons" in CSS
