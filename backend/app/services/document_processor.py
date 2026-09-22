"""
Document Processor — Pure Visual Document Ingestion.

Handles converting uploaded documents (PDF, JPG, PNG, DOCX) directly into high-fidelity
PIL Image objects and spatial dimension metadata for VLM/LLM visual understanding.
Zero OCR / ONNX dependencies.
"""
from __future__ import annotations
import os
from typing import List, Tuple, Dict, Any
from PIL import Image

from app.models.schemas import Block
from app.core.config import settings


class UnsupportedFileError(Exception):
    pass


def validate_file(filename: str, size_bytes: int) -> str:
    """Validates file extension and size limits against configured settings."""
    ext = os.path.splitext(filename.lower())[1]
    if ext not in settings.ALLOWED_EXTENSIONS:
        raise UnsupportedFileError(f"Unsupported file type: {ext}")
    if size_bytes > settings.MAX_UPLOAD_MB * 1024 * 1024:
        raise UnsupportedFileError(f"File exceeds {settings.MAX_UPLOAD_MB}MB limit")
    return ext


def _downscale_image(img: Image.Image, max_side: int) -> Image.Image:
    """
    Downscales image so its longest side is at most max_side pixels.
    Preserves aspect ratio. Returns original if already within bounds.
    """
    w, h = img.size
    if max(w, h) <= max_side:
        return img
    scale = max_side / max(w, h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return img.resize((new_w, new_h), Image.LANCZOS)


def render_pdf_pages(path: str, dpi: int | None = None) -> List[Image.Image]:
    """
    Renders PDF pages directly to high-fidelity PIL Images.
    Uses pypdfium2 (pure Python/C extension, zero external poppler binary required).
    DPI defaults to settings.PDF_RENDER_DPI (120) for faster processing.
    Large pages are downscaled to settings.VLM_IMAGE_MAX_SIDE.
    """
    render_dpi = dpi if dpi is not None else settings.PDF_RENDER_DPI
    max_side = settings.VLM_IMAGE_MAX_SIDE

    try:
        import pypdfium2 as pdfium
        pdf = pdfium.PdfDocument(path)
        images: List[Image.Image] = []
        for page in pdf:
            # scale=dpi/72 (e.g. 120/72 ≈ 1.67x scale for readable visual detail)
            bitmap = page.render(scale=render_dpi / 72.0)
            pil_image = bitmap.to_pil()
            if pil_image.mode != "RGB":
                pil_image = pil_image.convert("RGB")
            pil_image = _downscale_image(pil_image, max_side)
            images.append(pil_image)
        if images:
            return images
    except Exception as e:
        print(f"[DocumentProcessor] pypdfium2 rendering error ({e}), trying pdf2image fallback.")

    try:
        from pdf2image import convert_from_path
        raw_images = convert_from_path(path, dpi=render_dpi)
        rgb_images = [_downscale_image(img.convert("RGB") if img.mode != "RGB" else img, max_side) for img in raw_images]
        return rgb_images
    except Exception as e2:
        print(f"[DocumentProcessor] pdf2image fallback failed ({e2}).")
        raise RuntimeError("PDF rendering failed: Could not render PDF pages to images.") from e2


def render_docx_pages(path: str) -> List[Image.Image]:
    """
    Converts a .docx file to page images for VLM processing.
    Strategy: convert to PDF first (via docx2pdf or LibreOffice), then render pages.
    Falls back to extracting text as a rendered image if conversion tools unavailable.
    """
    import tempfile

    # Try docx2pdf (cross-platform, uses MS Word on Windows / LibreOffice on Linux)
    try:
        from docx2pdf import convert as docx2pdf_convert
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp_pdf_path = tmp.name
        docx2pdf_convert(path, tmp_pdf_path)
        images = render_pdf_pages(tmp_pdf_path)
        try:
            os.unlink(tmp_pdf_path)
        except Exception:
            pass
        if images:
            print(f"[DocumentProcessor] .docx converted via docx2pdf -> {len(images)} page(s).")
            return images
    except Exception as e:
        print(f"[DocumentProcessor] docx2pdf conversion failed ({e}), trying LibreOffice...")

    # Try LibreOffice headless (Linux/server environments)
    try:
        import subprocess
        import tempfile
        tmp_dir = tempfile.mkdtemp()
        result = subprocess.run(
            ["libreoffice", "--headless", "--convert-to", "pdf", "--outdir", tmp_dir, path],
            capture_output=True, timeout=60
        )
        if result.returncode == 0:
            base_name = os.path.splitext(os.path.basename(path))[0]
            tmp_pdf_path = os.path.join(tmp_dir, f"{base_name}.pdf")
            if os.path.exists(tmp_pdf_path):
                images = render_pdf_pages(tmp_pdf_path)
                try:
                    import shutil
                    shutil.rmtree(tmp_dir, ignore_errors=True)
                except Exception:
                    pass
                if images:
                    print(f"[DocumentProcessor] .docx converted via LibreOffice -> {len(images)} page(s).")
                    return images
    except Exception as e2:
        print(f"[DocumentProcessor] LibreOffice conversion failed ({e2}).")

    # Final fallback: render docx text content as a plain image
    try:
        from docx import Document as DocxDocument
        from PIL import ImageDraw, ImageFont
        doc = DocxDocument(path)
        full_text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        if not full_text:
            full_text = "[Empty or unreadable DOCX document]"

        # Render text onto a white A4-like image
        img_w, img_h = 1240, 1754
        img = Image.new("RGB", (img_w, img_h), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("arial.ttf", 18)
        except Exception:
            font = ImageFont.load_default()

        margin = 60
        x, y = margin, margin
        line_height = 22
        for line in full_text.split("\n"):
            if y + line_height > img_h - margin:
                break
            draw.text((x, y), line[:120], fill=(0, 0, 0), font=font)
            y += line_height

        print(f"[DocumentProcessor] .docx rendered as text image (fallback).")
        return [img]
    except Exception as e3:
        raise RuntimeError(f".docx processing failed entirely: {e3}") from e3


def render_document_images(path: str, ext: str) -> Tuple[int, List[Tuple[int, int]], Dict[int, Image.Image]]:
    """
    Renders any supported document format (.pdf, .png, .jpg, .jpeg, .docx) to high-fidelity page images.
    Enforces a MAX_PDF_PAGES cap: if the document exceeds the limit, pages are intelligently
    merged into composite images to ensure full coverage without exceeding VLM token limits.

    Returns:
      num_pages: int
      sizes: List of (width, height) per page
      page_images: Dict[page_num, PIL.Image]
    """
    page_images_dict: Dict[int, Image.Image] = {}
    max_side = settings.VLM_IMAGE_MAX_SIDE

    if ext.lower() == ".pdf":
        images = render_pdf_pages(path)
    elif ext.lower() == ".docx":
        images = render_docx_pages(path)
    else:
        img = Image.open(path).convert("RGB")
        img = _downscale_image(img, max_side)
        page_images_dict[1] = img
        return 1, [img.size], page_images_dict

    # Enforce page cap — merge excess pages into composite strips
    max_pages = settings.MAX_PDF_PAGES
    if len(images) > max_pages:
        print(f"[DocumentProcessor] Document has {len(images)} pages, exceeds MAX_PDF_PAGES={max_pages}. Merging to composites...")
        images = _merge_pages_to_composites(images, max_pages)

    sizes = [img.size for img in images]
    for idx, img in enumerate(images):
        page_images_dict[idx + 1] = img
    return len(images), sizes, page_images_dict


def _merge_pages_to_composites(images: List[Image.Image], target_count: int) -> List[Image.Image]:
    """
    Merges consecutive pages into vertical composite strips so the total count
    does not exceed target_count. Each composite preserves all content.
    """
    total = len(images)
    pages_per_composite = (total + target_count - 1) // target_count  # ceiling division
    composites: List[Image.Image] = []

    for i in range(0, total, pages_per_composite):
        chunk = images[i:i + pages_per_composite]
        if len(chunk) == 1:
            composites.append(chunk[0])
            continue

        max_width = max(img.width for img in chunk)
        total_height = sum(img.height for img in chunk)
        composite = Image.new("RGB", (max_width, total_height), color=(255, 255, 255))
        y_offset = 0
        for img in chunk:
            if img.width < max_width:
                # Center narrower pages
                x_offset = (max_width - img.width) // 2
                composite.paste(img, (x_offset, y_offset))
            else:
                composite.paste(img, (0, y_offset))
            y_offset += img.height

        # Downscale composite to keep payload manageable
        composite = _downscale_image(composite, settings.VLM_IMAGE_MAX_SIDE)
        composites.append(composite)

    print(f"[DocumentProcessor] Merged {total} pages into {len(composites)} composite images.")
    return composites


def process_document(path: str, ext: str, force_ocr: bool = False) -> Tuple[List[Block], int, List[Tuple[int, int]], Dict[int, Image.Image]]:
    """
    Backward-compatible pipeline entry point.
    Renders document to page images directly without running any OCR engine.
    Returns ([], num_pages, page_pixel_sizes, page_images_dict).
    """
    num_pages, sizes, page_images_dict = render_document_images(path, ext)
    blocks: List[Block] = []
    return blocks, num_pages, sizes, page_images_dict


def _get_ocr_engine():
    """Deprecated stub: OCR has been replaced by 100% VLM visual intelligence."""
    return None
