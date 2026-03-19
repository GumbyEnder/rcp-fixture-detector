"""Read fixture tags (e.g. L-200, QTY 3) near detected bounding boxes via PaddleOCR."""

import logging
import re
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Lazy-init OCR to avoid import cost when not needed
_ocr_instance = None


def _get_ocr(lang: str = "en"):
    global _ocr_instance
    if _ocr_instance is None:
        from paddleocr import PaddleOCR
        _ocr_instance = PaddleOCR(use_angle_cls=True, lang=lang, show_log=False)
    return _ocr_instance


def read_fixture_tags(
    image_path: str | Path,
    detections: list[dict],
    expand_ratio: float = 2.5,
    fixture_code_pattern: str = r"[A-Z]-\d{3}[A-Z]?",
    quantity_pattern: str = r"QTY\s*\d+",
    lang: str = "en",
) -> list[dict]:
    """Expand each detection bbox and OCR the surrounding region for fixture tags.

    Each detection dict must have ``bbox_abs`` = [x1, y1, x2, y2].
    Returns a copy of *detections* with ``fixture_code`` and ``quantity`` fields added.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        logger.warning("Cannot read image for OCR: %s", image_path)
        return detections

    img_h, img_w = img.shape[:2]
    ocr = _get_ocr(lang)
    code_re = re.compile(fixture_code_pattern)
    qty_re = re.compile(quantity_pattern, re.IGNORECASE)

    enriched = []
    for det in detections:
        det = dict(det)  # copy
        x1, y1, x2, y2 = det["bbox_abs"]
        bw = x2 - x1
        bh = y2 - y1
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        # Expand bbox
        ex1 = max(0, int(cx - bw * expand_ratio / 2))
        ey1 = max(0, int(cy - bh * expand_ratio / 2))
        ex2 = min(img_w, int(cx + bw * expand_ratio / 2))
        ey2 = min(img_h, int(cy + bh * expand_ratio / 2))

        crop = img[ey1:ey2, ex1:ex2]
        if crop.size == 0:
            enriched.append(det)
            continue

        result = ocr.ocr(crop, cls=True)
        texts = []
        if result and result[0]:
            texts = [line[1][0] for line in result[0]]

        full_text = " ".join(texts)

        code_match = code_re.search(full_text)
        qty_match = qty_re.search(full_text)

        det["fixture_code"] = code_match.group() if code_match else None
        det["quantity"] = int(re.search(r"\d+", qty_match.group()).group()) if qty_match else None
        det["ocr_text"] = full_text if texts else None

        enriched.append(det)

    n_codes = sum(1 for d in enriched if d.get("fixture_code"))
    logger.info("OCR: found fixture codes for %d/%d detections", n_codes, len(enriched))
    return enriched
