"""
Handles turning an uploaded file (PDF/PNG/JPG) into a normalized list of
Block objects: text + bbox + page + confidence.

Primary OCR engine: PaddleOCR (per plan section 9).
Native PDF text extraction remains for typed question papers.
Answer sheet processing ALWAYS uses image rendering + OCR to produce exact pixel bboxes.
"""
from __future__ import annotations
import io
import os
import uuid
from typing import List, Tuple
import numpy as np
from PIL import Image
import pypdf
from pdf2image import convert_from_path

from app.models.schemas import Block, BBox
from app.core.config import settings

# PaddleOCR lazy singleton
_ocr_engine = None
_ocr_failed = False


def _get_ocr_engine():
    global _ocr_engine, _ocr_failed
    if _ocr_engine is not None or _ocr_failed:
        return _ocr_engine
    try:
        from paddleocr import PaddleOCR
        # Initialize PaddleOCR for English
        _ocr_engine = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
    except Exception as e:
        print(f"[DocumentProcessor] PaddleOCR initialization failed ({e}), checking pytesseract fallback.")
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
    engine = _get_paddle_ocr()
    img_np = np.array(img.convert("RGB"))

    if engine is not None:
        try:
            # PaddleOCR returns list of lines: [ [ [ [x1,y1],[x2,y2],[x3,y3],[x4,y4] ], (text, confidence) ], ... ]
            results = engine.ocr(img_np, cls=True)
            blocks: List[Block] = []
            if results and results[0]:
                for line in results[0]:
                    box, (text, conf) = line
                    if not text or not text.strip():
                        continue
                    xs = [pt[0] for pt in box]
                    ys = [pt[1] for pt in box]
                    x_min, y_min = min(xs), min(ys)
                    x_max, y_max = max(xs), max(ys)

                    blocks.append(
                        Block(
                            id=str(uuid.uuid4())[:8],
                            text=text.strip(),
                            confidence=round(float(conf), 3),
                            bbox=BBox(
                                x=round(float(x_min), 1),
                                y=round(float(y_min), 1),
                                width=round(float(x_max - x_min), 1),
                                height=round(float(y_max - y_min), 1),
                            ),
                            page=page_num,
                            type="line",
                            source="paddleocr",
                        )
                    )
            return blocks
        except Exception as e:
            print(f"[DocumentProcessor] PaddleOCR execution error ({e}), trying pytesseract fallback.")

    # Pytesseract fallback if PaddleOCR is not installed/loading
    try:
        import pytesseract
        from pytesseract import Output
        data = pytesseract.image_to_data(img, output_type=Output.DICT)
        blocks: List[Block] = []
        n = len(data["text"])
        lines: dict = {}
        for i in range(n):
            word = data["text"][i].strip()
            if not word:
                continue
            conf = float(data["conf"][i]) if data["conf"][i] not in ("-1", -1) else 0.0
            key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
            x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
            if key not in lines:
                lines[key] = {"words": [], "x0": x, "y0": y, "x1": x + w, "y1": y + h, "confs": []}
            L = lines[key]
            L["words"].append(word)
            L["x0"] = min(L["x0"], x)
            L["y0"] = min(L["y0"], y)
            L["x1"] = max(L["x1"], x + w)
            L["y1"] = max(L["y1"], y + h)
            L["confs"].append(conf)

        for L in lines.values():
            text = " ".join(L["words"])
            avg_conf = sum(L["confs"]) / len(L["confs"]) / 100.0 if L["confs"] else 0.0
            blocks.append(
                Block(
                    id=str(uuid.uuid4())[:8],
                    text=text,
                    confidence=round(avg_conf, 3),
                    bbox=BBox(x=float(L["x0"]), y=float(L["y0"]), width=float(L["x1"] - L["x0"]), height=float(L["y1"] - L["y0"])),
                    page=page_num,
                    type="line",
                    source="ocr",
                )
            )
        return blocks
    except Exception as e:
        print(f"[DocumentProcessor] All OCR methods failed ({e}). Returning empty blocks.")
        return []


def _get_paddle_ocr():
    return _get_ocr_engine()


def _render_pdf_pages(path: str) -> List[Image.Image]:
    return convert_from_path(path, dpi=200)


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
