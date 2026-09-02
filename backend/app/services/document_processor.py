"""
Document Processor — Pure Visual Document Ingestion.

Handles converting uploaded documents (PDF, JPG, PNG) directly into high-fidelity
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


def render_pdf_pages(path: str, dpi: int = 150) -> List[Image.Image]:
    """
    Renders PDF pages directly to high-fidelity PIL Images.
    Uses pypdfium2 (pure Python/C extension, zero external poppler binary required).
    """
    try:
        import pypdfium2 as pdfium
        pdf = pdfium.PdfDocument(path)
        images: List[Image.Image] = []
        for page in pdf:
            # scale=dpi/72 (e.g. 150/72 = 2.083x scale for crisp visual detail)
            bitmap = page.render(scale=dpi / 72.0)
            pil_image = bitmap.to_pil()
            if pil_image.mode != "RGB":
                pil_image = pil_image.convert("RGB")
            images.append(pil_image)
        if images:
            return images
    except Exception as e:
        print(f"[DocumentProcessor] pypdfium2 rendering error ({e}), trying pdf2image fallback.")

    try:
        from pdf2image import convert_from_path
        raw_images = convert_from_path(path, dpi=dpi)
        rgb_images = [img.convert("RGB") if img.mode != "RGB" else img for img in raw_images]
        return rgb_images
    except Exception as e2:
        print(f"[DocumentProcessor] pdf2image fallback failed ({e2}).")
        raise RuntimeError("PDF rendering failed: Could not render PDF pages to images.") from e2


def render_document_images(path: str, ext: str) -> Tuple[int, List[Tuple[int, int]], Dict[int, Image.Image]]:
    """
    Renders any supported document format (.pdf, .png, .jpg, .jpeg) to high-fidelity page images.
    Returns:
      num_pages: int
      sizes: List of (width, height) per page
      page_images: Dict[page_num, PIL.Image]
    """
    page_images_dict: Dict[int, Image.Image] = {}
    if ext.lower() == ".pdf":
        images = render_pdf_pages(path, dpi=150)
        sizes = [img.size for img in images]
        for idx, img in enumerate(images):
            page_images_dict[idx + 1] = img
        return len(images), sizes, page_images_dict
    else:
        img = Image.open(path).convert("RGB")
        page_images_dict[1] = img
        return 1, [img.size], page_images_dict


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
