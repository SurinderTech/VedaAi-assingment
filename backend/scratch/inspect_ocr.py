import sys
import os
from PIL import Image
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.services.document_processor import _ocr_image, _get_ocr_engine

targets = ["qp.png", "qp8.png", "ans.png", "ans8.png"]

for t in targets:
    img_path = os.path.abspath(os.path.join(os.path.dirname(__file__), t))
    if not os.path.exists(img_path):
        continue
    img = Image.open(img_path)
    print(f"\n==================== {t} ====================")
    print("Image dimensions:", img.size, img.mode)

    engine = _get_ocr_engine()
    res = engine(np.array(img.convert("RGB")))
    blocks = _ocr_image(img, page_num=1)
    print(f"Extracted Blocks ({len(blocks)}):")
    for b in blocks:
        print(f"  ID: {b.id} | BBox: [{b.bbox.x}, {b.bbox.y}, {b.bbox.width}, {b.bbox.height}] | Conf: {b.confidence} | Text: '{b.text}'")

