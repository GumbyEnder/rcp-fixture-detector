"""Template-based ceiling fan detection using cv2.matchTemplate.

Replaces the Hough circle + blade line approach which produced too many false
positives on architectural drawings. Uses a reference fan symbol extracted from
a known RCP drawing and does multi-scale normalized cross-correlation.

Usage:
    from rcp_detector.detection.template_fan_detector import detect_fans_template
    fans = detect_fans_template(page_image_gray)
"""

import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_TEMPLATE_DIR = Path(__file__).parent / "templates"
_DEFAULT_TEMPLATE = _TEMPLATE_DIR / "ceiling_fan.png"


def detect_fans_template(
    image: np.ndarray,
    template_path: str | Path | None = None,
    scales: tuple[float, ...] = (0.85, 1.0, 1.15),
    threshold: float = 0.65,
    nms_dist: float = 100,
) -> list[dict]:
    """Detect ceiling fan symbols via multi-scale template matching.

    Args:
        image: Full-page image (BGR or grayscale).
        template_path: Path to grayscale fan template PNG. Uses built-in if None.
        scales: Template scale factors to try (handles slight size variation).
        threshold: Minimum TM_CCOEFF_NORMED score to count as a match.
        nms_dist: Minimum center-to-center distance (px) to keep both detections.

    Returns:
        List of dicts with keys: center_x, center_y, width, height, confidence, scale.
    """
    if template_path is None:
        template_path = _DEFAULT_TEMPLATE
    template_path = Path(template_path)

    if not template_path.exists():
        logger.warning("Fan template not found at %s — skipping fan detection", template_path)
        return []

    template = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE)
    if template is None:
        logger.warning("Could not read fan template at %s", template_path)
        return []

    # Convert image to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    raw_matches: list[tuple[int, int, int, int, float, float]] = []

    for scale in scales:
        w = int(template.shape[1] * scale)
        h = int(template.shape[0] * scale)
        if w < 30 or h < 30 or w > gray.shape[1] or h > gray.shape[0]:
            continue

        scaled_tmpl = cv2.resize(template, (w, h))
        result = cv2.matchTemplate(gray, scaled_tmpl, cv2.TM_CCOEFF_NORMED)

        locs = np.where(result >= threshold)
        for pt_y, pt_x in zip(*locs):
            score = float(result[pt_y, pt_x])
            raw_matches.append((pt_x, pt_y, w, h, score, scale))

    if not raw_matches:
        logger.info("Fan template matching: 0 raw matches (threshold=%.2f)", threshold)
        return []

    # NMS: keep highest-scoring match per location
    raw_matches.sort(key=lambda r: r[4], reverse=True)
    kept: list[dict] = []

    for x, y, w, h, score, scale in raw_matches:
        cx = x + w // 2
        cy = y + h // 2
        is_dup = False
        for existing in kept:
            if (abs(cx - existing["center_x"]) < nms_dist
                    and abs(cy - existing["center_y"]) < nms_dist):
                is_dup = True
                break
        if not is_dup:
            kept.append({
                "center_x": cx,
                "center_y": cy,
                "width": w,
                "height": h,
                "confidence": score,
                "scale": scale,
            })

    logger.info("Fan template matching: %d raw → %d unique (threshold=%.2f)",
                len(raw_matches), len(kept), threshold)
    return kept
