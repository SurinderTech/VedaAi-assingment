"""
Handles turning an uploaded file (PDF/PNG/JPG) into a normalized list of
Block objects: text + bbox + page + confidence.

Primary OCR engine: RapidOCR (Official PaddleOCR ONNX C++ engine).
Native PDF text extraction remains for typed question papers.
Answer sheet processing ALWAYS uses image rendering + RapidOCR to produce exact pixel bboxes.
Uses pypdfium2 for zero-dependency Windows PDF image rendering.
"""
from __future__ import annotations
import io
import os
import uuid
from typing import List, Tuple
import numpy as np
from PIL import Image
import pypdf

from app.models.schemas import Block, BBox
from app.core.config import settings

# RapidOCR / PaddleOCR lazy singleton
_ocr_engine = None
_ocr_failed = False


def _get_ocr_engine():
    global _ocr_engine, _ocr_failed
    if _ocr_engine is not None or _ocr_failed:
        return _ocr_engine
    try:
        from rapidocr_onnxruntime import RapidOCR
        _ocr_engine = RapidOCR()
        print("[DocumentProcessor] RapidOCR (PaddleOCR ONNX) initialized successfully.")
    except Exception as e:
        print(f"[DocumentProcessor] RapidOCR initialization failed ({e}), trying paddleocr fallback.")
        try:
            from paddleocr import PaddleOCR
            _ocr_engine = PaddleOCR(lang="en")
        except Exception as e2:
            print(f"[DocumentProcessor] PaddleOCR fallback failed ({e2}).")
            _ocr_failed = True
            _ocr_engine = None
    return _ocr_engine


class UnsupportedFileError(Exception):
    pass


def validate_file(filename: str, size_bytes: int) -> str:
    ext = os.path.splitext(filename.lower())[1]
    if ext not in settings.ALLOWED_EXTENSIONS:
        raise UnsupportedFileError(f"Unsupported file type: {ext}")
    if size_bytes > settings.MAX_UPLOAD_MB * 1024 * 1024:
        raise UnsupportedFileError(f"File exceeds {settings.MAX_UPLOAD_MB}MB limit")
    return ext


def _pdf_has_native_text(path: str) -> bool:
    try:
        reader = pypdf.PdfReader(path)
        chars = 0
        for page in reader.pages[:3]:
            chars += len((page.extract_text() or "").strip())
        return chars > 40
    except Exception:
        return False


def _extract_native_pdf_blocks(path: str) -> Tuple[List[Block], int]:
    reader = pypdf.PdfReader(path)
    blocks: List[Block] = []
    num_pages = len(reader.pages)
    for page_idx, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        media = page.mediabox
        page_w = float(media.width)
        page_h = float(media.height)
        lines = [l for l in text.split("\n") if l.strip()]
        if not lines:
            continue
        line_h = page_h / max(len(lines), 1)
        for i, line in enumerate(lines):
            blocks.append(
                Block(
                    id=str(uuid.uuid4())[:8],
                    text=line.strip(),
                    confidence=0.99,
                    bbox=BBox(x=0.0, y=round(i * line_h, 1), width=round(page_w, 1), height=round(line_h, 1)),
                    page=page_idx + 1,
                    type="line",
                    source="native_pdf",
                )
            )
    return blocks, num_pages


def _ocr_image(img: Image.Image, page_num: int) -> List[Block]:
    engine = _get_ocr_engine()
    if engine is None:
        print("[DocumentProcessor] OCR engine unavailable.")
        return []

    img_np = np.array(img.convert("RGB"))
    try:
        # RapidOCR returns (results, elapse_time) where results is a list of [box, text, confidence]
        res = engine(img_np)
        results = res[0] if isinstance(res, tuple) else res
        blocks: List[Block] = []

        if results:
            for line in results:
                if not line or len(line) < 3:
                    # Check PaddleOCR fallback format [box, (text, conf)]
                    if isinstance(line, (list, tuple)) and len(line) == 2 and isinstance(line[1], (list, tuple)):
                        box, (text, conf) = line[0], line[1]
                    else:
                        continue
                else:
                    box, text, conf = line[0], line[1], line[2]

                if not text or not str(text).strip():
                    continue

                xs = [pt[0] for pt in box]
                ys = [pt[1] for pt in box]
                x_min, y_min = min(xs), min(ys)
                x_max, y_max = max(xs), max(ys)

                blocks.append(
                    Block(
                        id=str(uuid.uuid4())[:8],
                        text=str(text).strip(),
                        confidence=round(float(conf), 3),
                        bbox=BBox(
                            x=round(float(x_min), 1),
                            y=round(float(y_min), 1),
                            width=round(float(x_max - x_min), 1),
                            height=round(float(y_max - y_min), 1),
                        ),
                        page=page_num,
                        type="line",
                        source="ocr",
                    )
                )
        return blocks
    except Exception as e:
        print(f"[DocumentProcessor] OCR execution error on page {page_num}: {e}")
        return []


def _render_pdf_pages(path: str) -> List[Image.Image]:
    """
    Renders PDF pages to PIL images using pypdfium2 (pure Python/C extension, zero external poppler binary required).
    """
    try:
        import pypdfium2 as pdfium
        pdf = pdfium.PdfDocument(path)
        images = []
        for page in pdf:
            bitmap = page.render(scale=200 / 72)
            pil_image = bitmap.to_pil()
            images.append(pil_image)
        if images:
            return images
    except Exception as e:
        print(f"[DocumentProcessor] pypdfium2 rendering failed ({e}), trying pdf2image fallback.")

    try:
        from pdf2image import convert_from_path
        return convert_from_path(path, dpi=200)
    except Exception as e:
        print(f"[DocumentProcessor] pdf2image rendering failed ({e}).")
        raise RuntimeError("PDF rendering failed: Could not render PDF pages to images.")


def process_document(path: str, ext: str, force_ocr: bool = False) -> Tuple[List[Block], int, List[Tuple[int, int]]]:
    """
    Returns (blocks, num_pages, page_pixel_sizes[(w,h) per page]).
    If force_ocr=True (e.g. for answer sheets), always renders pages to images and runs OCR
    so bounding boxes are exact image pixel coordinates.
    """
    if ext == ".pdf":
        if not force_ocr and _pdf_has_native_text(path):
            blocks, num_pages = _extract_native_pdf_blocks(path)
            reader = pypdf.PdfReader(path)
            sizes = [(int(p.mediabox.width), int(p.mediabox.height)) for p in reader.pages]
            return blocks, num_pages, sizes
        else:
            images = _render_pdf_pages(path)
            blocks: List[Block] = []
            sizes = []
            for idx, img in enumerate(images):
                blocks.extend(_ocr_image(img, idx + 1))
                sizes.append(img.size)
            return blocks, len(images), sizes
    else:
        img = Image.open(path).convert("RGB")
        blocks = _ocr_image(img, 1)
        return blocks, 1, [img.size]
