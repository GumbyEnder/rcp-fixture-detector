"""Ceiling fan detection via Hough circles + blade line verification.

Ceiling fans on RCP drawings appear as circles with 3-5 internal radial lines (blades).
This module uses classical CV — no ML training required.
"""

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def detect_ceiling_fans(
    image: np.ndarray,
    min_radius: int = 8,
    max_radius: int = 25,
    min_blade_lines: int = 3,
    hough_param1: int = 50,
    hough_param2: int = 25,
) -> list[dict]:
    """Detect ceiling fan symbols in a tile image.

    Fan symbols are circles with internal radial lines (blades).

    Returns list of dicts with keys: center_x, center_y, radius, confidence, blade_count.
    """
    if image is None or image.size == 0:
        return []

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.5)

    # Detect circles
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max_radius * 2,
        param1=hough_param1,
        param2=hough_param2,
        minRadius=min_radius,
        maxRadius=max_radius,
    )

    if circles is None:
        return []

    # Edge image for blade line detection
    edges = cv2.Canny(gray, 50, 150)

    results = []
    for circle in circles[0]:
        cx, cy, r = int(circle[0]), int(circle[1]), int(circle[2])

        # Verify blade lines inside the circle
        blade_count = _count_blade_lines(edges, cx, cy, r)

        if blade_count >= min_blade_lines:
            # Confidence based on blade count (3 blades = 0.6, 4 = 0.8, 5+ = 0.95)
            conf = min(0.5 + blade_count * 0.15, 0.95)
            results.append({
                "center_x": cx,
                "center_y": cy,
                "radius": r,
                "confidence": conf,
                "blade_count": blade_count,
            })

    return results


def _count_blade_lines(
    edges: np.ndarray,
    cx: int,
    cy: int,
    radius: int,
) -> int:
    """Count line segments passing through/near the center of a circle."""
    h, w = edges.shape
    margin = 4

    # Crop region around the circle (with padding)
    pad = radius + margin
    x1 = max(0, cx - pad)
    y1 = max(0, cy - pad)
    x2 = min(w, cx + pad)
    y2 = min(h, cy + pad)

    crop = edges[y1:y2, x1:x2]
    if crop.size == 0:
        return 0

    # Mask to only look inside the circle
    mask = np.zeros_like(crop)
    local_cx = cx - x1
    local_cy = cy - y1
    cv2.circle(mask, (local_cx, local_cy), radius - 2, 255, -1)
    masked = cv2.bitwise_and(crop, mask)

    # Detect line segments inside the circle
    lines = cv2.HoughLinesP(
        masked,
        rho=1,
        theta=np.pi / 180,
        threshold=8,
        minLineLength=max(radius * 0.4, 3),
        maxLineGap=3,
    )

    if lines is None:
        return 0

    # Count lines that pass near the center
    center_tolerance = radius * 0.4
    blade_angles: list[float] = []

    for line in lines:
        lx1, ly1, lx2, ly2 = line[0]

        # Distance from center to line segment
        dist = _point_to_segment_dist(local_cx, local_cy, lx1, ly1, lx2, ly2)
        if dist > center_tolerance:
            continue

        # Compute angle to avoid counting the same blade twice
        angle = np.degrees(np.arctan2(ly2 - ly1, lx2 - lx1)) % 180
        is_duplicate = False
        for existing_angle in blade_angles:
            if abs(angle - existing_angle) < 15 or abs(angle - existing_angle - 180) < 15:
                is_duplicate = True
                break
        if not is_duplicate:
            blade_angles.append(angle)

    return len(blade_angles)


def _point_to_segment_dist(
    px: float, py: float,
    x1: float, y1: float,
    x2: float, y2: float,
) -> float:
    """Distance from point (px, py) to line segment (x1,y1)-(x2,y2)."""
    dx = x2 - x1
    dy = y2 - y1
    len_sq = dx * dx + dy * dy
    if len_sq == 0:
        return ((px - x1) ** 2 + (py - y1) ** 2) ** 0.5

    t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / len_sq))
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    return ((px - proj_x) ** 2 + (py - proj_y) ** 2) ** 0.5


def dedup_fan_detections(
    detections: list[dict],
    merge_radius: float = 50,
) -> list[dict]:
    """Merge duplicate fan detections from overlapping tiles.

    Two detections are the same fan if their page-level centers are within *merge_radius*.
    """
    if not detections:
        return []

    # Sort by confidence descending
    sorted_dets = sorted(detections, key=lambda d: d["confidence"], reverse=True)
    kept = []

    for det in sorted_dets:
        is_dup = False
        for existing in kept:
            dist = (
                (det["center_x"] - existing["center_x"]) ** 2
                + (det["center_y"] - existing["center_y"]) ** 2
            ) ** 0.5
            if dist < merge_radius:
                is_dup = True
                break
        if not is_dup:
            kept.append(det)

    return kept
