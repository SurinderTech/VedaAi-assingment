"""
Inspect Question 4 and Question 5 OCR blocks on Page 6 and Page 7.
"""
import sys, os
from pathlib import Path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.services.document_processor import process_document

pdf_path = Path("C:/Users/surin/AppData/Local/Temp/vedaai_uploads/8a27cbab5a31/question_paper.pdf")
blocks, num_pages, page_sizes, page_images = process_document(str(pdf_path), ".pdf", force_ocr=False)

p6_blocks = [b for b in blocks if b.page == 6]
p7_blocks = [b for b in blocks if b.page == 7]

print("=== PAGE 6 OCR BLOCKS BETWEEN Y=1100 AND Y=1400 (Around Q4) ===")
for b in sorted(p6_blocks, key=lambda x: (x.bbox.y, x.bbox.x)):
    if 1100 <= b.bbox.y <= 1400:
        print(f"  [{b.id}] bbox=({b.bbox.x:4.0f},{b.bbox.y:4.0f},{b.bbox.width:4.0f},{b.bbox.height:4.0f}) text={b.text!r}")

print("\n=== PAGE 7 FIRST 15 OCR BLOCKS ===")
for b in sorted(p7_blocks, key=lambda x: (x.bbox.y, x.bbox.x))[:15]:
    print(f"  [{b.id}] bbox=({b.bbox.x:4.0f},{b.bbox.y:4.0f},{b.bbox.width:4.0f},{b.bbox.height:4.0f}) text={b.text!r}")
