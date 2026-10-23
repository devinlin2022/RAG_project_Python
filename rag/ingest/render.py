"""Render a PDF page to PNG bytes for OCR / vision calls."""

import io
import logging

log = logging.getLogger(__name__)


def render_page(path: str, page_no: int, dpi: int = 150) -> bytes:
    """Render one page (1-based) to PNG. Raises ValueError on failure."""
    try:
        import pypdfium2 as pdfium
    except ImportError as exc:
        raise ValueError(
            "缺少 pypdfium2，无法把页面转成图片：pip install pypdfium2"
        ) from exc

    doc = None
    try:
        doc = pdfium.PdfDocument(str(path))
        if not 1 <= page_no <= len(doc):
            raise ValueError(f"页码 {page_no} 超出范围（共 {len(doc)} 页）")
        page = doc[page_no - 1]
        # scale = dpi / 72 because PDF user units are 1/72 inch
        bitmap = page.render(scale=dpi / 72)
        image = bitmap.to_pil()
        buffer = io.BytesIO()
        image.save(buffer, format="PNG", optimize=True)
        return buffer.getvalue()
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"第 {page_no} 页渲染失败：{exc}") from exc
    finally:
        if doc is not None:
            doc.close()


def image_block(png: bytes) -> dict:
    """OpenAI-compatible vision message part (base64 data URL)."""
    import base64

    payload = base64.b64encode(png).decode("ascii")
    return {
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,{payload}"},
    }
