"""
1. Tile-based OCR with spatial deduplication (fixes overlap inflation)
2. Ceiling fan detection via Hough circles (Phase 2)
3. Schedule table parsing from legend region
4. Partial read filtering (L-10 → L-100)
5. Configurable DPI (already wired in CLI)
"""

import logging
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import time

logger = logging.getLogger(__name__)

# Lazy-init to avoid import cost
_ocr_instance = None


def _get_ocr(lang: str = "en"):
    global _ocr_instance
    if _ocr_instance is None:
        from paddleocr import PaddleOCR
        _ocr_instance = PaddleOCR(use_angle_cls=True, lang=lang, show_log=False)
    return _ocr_instance


@lru_cache(maxsize=128)
def _ocr_page_result(image_path: str, lang: str = "en"):
    return _get_ocr(lang).ocr(str(image_path))


def _parse_schedule_entries(lines) -> dict[str, str]:
    schedule: dict[str, str] = {}
    for line in lines or []:
        payload = line[1]
        text = payload[0] if isinstance(payload, (list, tuple)) else payload
        match = SCHEDULE_LINE_RE.search(text)
        if match:
            code = match.group(1)
            desc = match.group(2).strip()
            if _is_fixture_code(code) and len(desc) > 2:
                schedule[code] = desc
    return schedule


# ── Regex patterns ───────────────────────────────────────────────────────────

FIXTURE_CODE_RE = re.compile(r"\b([A-Z]{1,3}-\d{1,4}[A-Z]?)\b")
QTY_EXPLICIT_RE = re.compile(r"QTY\.?\s*:?\s*(\d+)", re.IGNORECASE)
QTY_PAREN_RE = re.compile(r"\((\d{1,3})\)")
SCHEDULE_LINE_RE = re.compile(r"\b([A-Z]{1,3}-\d{1,4}[A-Z]?)\b\s+(.+)", re.IGNORECASE)

# Codes to always skip (sheet IDs, not fixtures)
SKIP_CODES = {"ID-2", "ID-8", "A-1", "A-2", "A-3", "ID-1", "ID-3", "ID-4", "ID-5",
              "ID-6", "ID-7", "ID-9", "ID-10"}


@dataclass
class FixtureOccurrence:
    """A single occurrence of a fixture code found in the drawing."""
    code: str
    quantity: int
    bbox: list  # [x1, y1, x2, y2] in PAGE coordinates
    raw_text: str
    confidence: float = 0.0
    source: str = "plan"  # "plan", "schedule", or "fan"


@dataclass
class FixtureCountResult:
    """Aggregated results for one page."""
    page_name: str
    fixture_counts: dict = field(default_factory=dict)
    schedule_entries: dict = field(default_factory=dict)
    occurrences: list = field(default_factory=list)
    fan_count: int = 0
    all_ocr_texts: list = field(default_factory=list)
    metrics: dict = field(default_factory=dict)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _bbox_from_points(points: list) -> list[float]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return [min(xs), min(ys), max(xs), max(ys)]


def _iou(a: list[float], b: list[float]) -> float:
    """Intersection over union for two [x1,y1,x2,y2] boxes."""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def _center_dist(a: list[float], b: list[float]) -> float:
    """Distance between centers of two bboxes."""
    acx, acy = (a[0] + a[2]) / 2, (a[1] + a[3]) / 2
    bcx, bcy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    return ((acx - bcx) ** 2 + (acy - bcy) ** 2) ** 0.5


def _is_fixture_code(code: str) -> bool:
    """Check if a code looks like a real fixture code (not a sheet ID or spec)."""
    if code in SKIP_CODES:
        return False
    # Single digit after dash AND prefix not a known fixture prefix → skip
    if re.match(r"^[A-Z]-\d$", code) and code[0] not in ("L", "C", "S", "E", "P", "F"):
        return False
    return True


# ── Improvement 4: Partial read filtering ────────────────────────────────────


def _filter_partial_reads(occurrences: list[FixtureOccurrence]) -> list[FixtureOccurrence]:
    """Remove fixture codes that are clearly truncated reads of longer codes.

    E.g., if both L-10 and L-100 exist, drop L-10 (partial digit read).

    But L-100 is NOT a partial of L-100A — suffix letters indicate a distinct code.
    Only filter when the longer code extends the numeric part, not when it adds
    a letter suffix.
    """
    all_codes = {occ.code for occ in occurrences}

    def is_partial(code: str) -> bool:
        for other in all_codes:
            if other == code:
                continue
            if not other.startswith(code):
                continue
            if len(other) <= len(code):
                continue
            # The extra characters after `code` — if they're digits, it's a partial
            # digit read (L-10 → L-100). If they're letters, it's a variant suffix
            # (L-100 → L-100A) and the shorter code is legitimate.
            suffix = other[len(code):]
            if suffix[0].isdigit():
                return True
            # suffix starts with a letter → code is a real base code, not partial
        return False

    partial_codes = {c for c in all_codes if is_partial(c)}
    if partial_codes:
        logger.info("Filtering partial reads: %s", partial_codes)

    return [occ for occ in occurrences if occ.code not in partial_codes]


# ── Improvement 1: Spatial deduplication ─────────────────────────────────────


def _spatial_dedup(
    occurrences: list[FixtureOccurrence],
    iou_threshold: float = 0.3,
    center_dist_threshold: float = 40.0,
) -> list[FixtureOccurrence]:
    """Remove duplicate occurrences of the same fixture code at overlapping locations.

    Two occurrences are duplicates if they have the same code AND either:
    - IoU > threshold, OR
    - center distance < threshold (for very small text boxes)
    """
    by_code: dict[str, list[FixtureOccurrence]] = defaultdict(list)
    for occ in occurrences:
        by_code[occ.code].append(occ)

    deduped: list[FixtureOccurrence] = []

    for code, group in by_code.items():
        # Sort by confidence descending — keep highest confidence version
        group.sort(key=lambda o: o.confidence, reverse=True)
        kept: list[FixtureOccurrence] = []

        for occ in group:
            is_dup = False
            for existing in kept:
                if (_iou(occ.bbox, existing.bbox) > iou_threshold
                        or _center_dist(occ.bbox, existing.bbox) < center_dist_threshold):
                    is_dup = True
                    # Propagate quantity if the duplicate had a QTY tag
                    if occ.quantity > 1 and existing.quantity == 1:
                        existing.quantity = occ.quantity
                    break
            if not is_dup:
                kept.append(occ)

        deduped.extend(kept)

    return deduped


# ── Improvement 3: Schedule table parsing ────────────────────────────────────


def _extract_schedule(
    image: np.ndarray,
    lang: str = "en",
) -> dict[str, str]:
    """OCR the schedule/legend region of the drawing and extract fixture descriptions.

    Crops the right 30% and bottom 20% strips, OCRs them, and parses fixture code + description.
    """
    h, w = image.shape[:2]
    ocr = _get_ocr(lang)
    schedule: dict[str, str] = {}

    regions = [
        ("right", image[:, int(w * 0.72):]),           # right 28%
        ("bottom", image[int(h * 0.82):, :]),           # bottom 18%
    ]

    for region_name, crop in regions:
        if crop.size == 0:
            continue

        result = ocr.ocr(crop)
        if not result or not result[0]:
            continue

        schedule.update(_parse_schedule_entries(result[0]))

    if schedule:
        logger.info("Schedule entries found: %s", list(schedule.keys()))

    return schedule


# ── Core: Tile-based OCR ─────────────────────────────────────────────────────


def _find_qty_tags_in_tile(
    all_texts: list[tuple[list, str, float]],
) -> list[tuple[list[float], int]]:
    """Extract all QTY tags from a tile's OCR results.

    Returns list of (bbox, quantity) tuples.
    """
    qty_tags = []
    for points, text, _conf in all_texts:
        bbox = _bbox_from_points(points)
        qty_match = QTY_EXPLICIT_RE.search(text)
        if qty_match:
            qty_tags.append((bbox, int(qty_match.group(1))))
            continue
        paren_match = QTY_PAREN_RE.search(text)
        if paren_match:
            val = int(paren_match.group(1))
            if 1 <= val <= 99:
                qty_tags.append((bbox, val))
    return qty_tags


def _assign_qty_1to1(
    fixture_bboxes: list[tuple[str, list[float]]],
    qty_tags: list[tuple[list[float], int]],
    search_radius: float = 200,
) -> dict[int, int]:
    """1:1 nearest-neighbor assignment of QTY tags to fixture codes.

    Each QTY tag is assigned to at most one fixture (the nearest one within
    search_radius). Each fixture gets at most one QTY tag.

    Returns mapping of fixture_index -> quantity.
    """
    if not qty_tags or not fixture_bboxes:
        return {}

    # Build distance matrix: (dist, fixture_idx, qty_idx)
    pairs = []
    for fi, (_code, fbbox) in enumerate(fixture_bboxes):
        fcx = (fbbox[0] + fbbox[2]) / 2
        fcy = (fbbox[1] + fbbox[3]) / 2
        for qi, (qbbox, _qty) in enumerate(qty_tags):
            qcx = (qbbox[0] + qbbox[2]) / 2
            qcy = (qbbox[1] + qbbox[3]) / 2
            dist = ((fcx - qcx) ** 2 + (fcy - qcy) ** 2) ** 0.5
            if dist <= search_radius:
                pairs.append((dist, fi, qi))

    # Greedy 1:1 matching: sort by distance, assign closest pairs first
    pairs.sort()
    assigned_fixtures: set[int] = set()
    assigned_qtys: set[int] = set()
    result: dict[int, int] = {}

    for dist, fi, qi in pairs:
        if fi in assigned_fixtures or qi in assigned_qtys:
            continue
        result[fi] = qty_tags[qi][1]
        assigned_fixtures.add(fi)
        assigned_qtys.add(qi)

    return result


def _find_nearby_quantity_in_tile(
    fixture_bbox: list[float],
    all_texts: list[tuple[list, str, float]],
    search_radius: float = 300,
) -> int:
    """Find the nearest quantity tag to a fixture bbox in a full-page OCR result.

    Used by the legacy full-page OCR path. Returns 0 if no nearby quantity is found.
    """
    qty_tags = _find_qty_tags_in_tile(all_texts)
    if not qty_tags:
        return 0

    fcx = (fixture_bbox[0] + fixture_bbox[2]) / 2
    fcy = (fixture_bbox[1] + fixture_bbox[3]) / 2
    best_qty = 0
    best_dist = None

    for qbbox, qty in qty_tags:
        qcx = (qbbox[0] + qbbox[2]) / 2
        qcy = (qbbox[1] + qbbox[3]) / 2
        dist = ((fcx - qcx) ** 2 + (fcy - qcy) ** 2) ** 0.5
        if dist <= search_radius and (best_dist is None or dist < best_dist):
            best_dist = dist
            best_qty = qty

    return best_qty


def count_fixtures_tiled(
    image_path: str | Path,
    patch_size: int = 640,
    overlap_factor: float = 0.5,
    lang: str = "en",
    search_radius: float = 350,
    dedup_iou: float = 0.3,
    dedup_dist: float = 40.0,
    detect_fans: bool = False,  # experimental — high FP rate on architectural drawings
    fan_min_radius: int = 8,
    fan_max_radius: int = 25,
    dpi: int = 300,
) -> FixtureCountResult:
    """Tile-based OCR with spatial dedup, partial read filtering, and fan detection.

    This is the main entry point for fixture counting on a single page image.
    """
    image_path = Path(image_path)
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    img_h, img_w = img.shape[:2]
    ocr = _get_ocr(lang)

    step = max(1, int(patch_size * (1 - overlap_factor)))
    all_raw_texts: list[str] = []
    all_occurrences: list[FixtureOccurrence] = []
    fan_detections: list[dict] = []
    started = time.monotonic()
    logger.info("%s: starting tiled OCR (%dx%d px, patch=%d, overlap=%.0f%%)", image_path.name, img_w, img_h, patch_size, overlap_factor * 100)

    # ── Step 1: Extract schedule from legend area first ──
    schedule_start = time.monotonic()
    schedule = _extract_schedule(img, lang)
    logger.info("%s: schedule extraction finished in %.1fs (%d entries)", image_path.name, time.monotonic() - schedule_start, len(schedule))

    # ── Step 2: Tile and OCR ──
    tile_count = 0
    skipped_blank = 0
    y = 0
    while y + patch_size <= img_h:
        x = 0
        while x + patch_size <= img_w:
            tile = img[y:y + patch_size, x:x + patch_size]

            # Skip completely blank tiles (>99% near-white pixels) — performance win
            gray_tile = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY) if len(tile.shape) == 3 else tile
            white_ratio = np.mean(gray_tile > 250)
            if white_ratio > 0.99:
                skipped_blank += 1
                x += step
                continue

            # OCR this tile
            result = ocr.ocr(tile)
            if result and result[0]:
                tile_texts: list[tuple[list, str, float]] = []
                for line in result[0]:
                    points = line[0]
                    text = line[1][0]
                    conf = line[1][1]
                    if conf > 0.4:
                        tile_texts.append((points, text, conf))
                        all_raw_texts.append(text)

                # Extract fixture codes from this tile (two-pass for 1:1 QTY matching)
                tile_fixtures: list[tuple[str, list[float], str, float]] = []
                for points, text, conf in tile_texts:
                    local_bbox = _bbox_from_points(points)
                    codes = FIXTURE_CODE_RE.findall(text)

                    for code in codes:
                        if not _is_fixture_code(code):
                            continue
                        tile_fixtures.append((code, local_bbox, text, conf))

                # 1:1 nearest-neighbor QTY assignment for this tile
                qty_tags = _find_qty_tags_in_tile(tile_texts)
                fixture_bboxes = [(code, bbox) for code, bbox, _text, _conf in tile_fixtures]
                qty_map = _assign_qty_1to1(fixture_bboxes, qty_tags, search_radius)

                for fi, (code, local_bbox, text, conf) in enumerate(tile_fixtures):
                    page_bbox = [
                        local_bbox[0] + x,
                        local_bbox[1] + y,
                        local_bbox[2] + x,
                        local_bbox[3] + y,
                    ]
                    qty = qty_map.get(fi, 1)

                    all_occurrences.append(FixtureOccurrence(
                        code=code,
                        quantity=qty,
                        bbox=page_bbox,
                        raw_text=text,
                        confidence=conf,
                        source="plan",
                    ))

            # Note: Fan detection moved to Step 6 — template matching on full page
            # is much more accurate than per-tile Hough circles.

            tile_count += 1
            x += step
        y += step

    logger.info("OCR scanned %d tiles (%d blank skipped), found %d raw occurrences",
                tile_count, skipped_blank, len(all_occurrences))

    raw_occurrence_count = len(all_occurrences)

    # ── Step 4: Filter partial reads ──
    all_occurrences = _filter_partial_reads(all_occurrences)

    # ── Step 5: Spatial deduplication ──
    deduped = _spatial_dedup(all_occurrences, iou_threshold=dedup_iou, center_dist_threshold=dedup_dist)
    logger.info("After dedup: %d unique occurrences (was %d)", len(deduped), len(all_occurrences))

    # ── Step 6: Fan detection (template-based on full page, or Hough per-tile) ──
    fan_count = 0
    fan_detection_method = "disabled"
    if detect_fans:
        try:
            from rcp_detector.detection.template_fan_detector import detect_fans_template
            nms = int(100 * dpi / 300)  # scale NMS distance with DPI
            template_fans = detect_fans_template(img, threshold=0.80, dpi=dpi, nms_dist=nms)
            fan_count = len(template_fans)
            fan_detection_method = "template"
            logger.info("Template fan detection: %d fans found", fan_count)
        except Exception as e:
            logger.warning("Template fan detection failed (%s), falling back to Hough", e)
            if fan_detections:
                from rcp_detector.detection.fan_detector import dedup_fan_detections
                unique_fans = dedup_fan_detections(fan_detections, merge_radius=fan_max_radius * 4)
                fan_count = len(unique_fans)
                fan_detection_method = "hough"
                logger.info("Hough fan detection fallback: %d fans", fan_count)
    elif fan_detections:
        from rcp_detector.detection.fan_detector import dedup_fan_detections
        unique_fans = dedup_fan_detections(fan_detections, merge_radius=fan_max_radius * 4)
        fan_count = len(unique_fans)
        logger.info("Fans detected: %d (after dedup from %d raw)", fan_count, len(fan_detections))

    # ── Step 7: Aggregate counts ──
    fixture_counts: dict[str, int] = defaultdict(int)
    for occ in deduped:
        if occ.source == "plan":
            fixture_counts[occ.code] += occ.quantity

    # Add fan count
    if fan_count > 0:
        fixture_counts["CEILING_FAN"] = fan_count

    # Note schedule-only codes with 0
    for code in schedule:
        if code not in fixture_counts:
            fixture_counts[code] = 0

    # Filter out non-fixture codes (P-*, WC-*, etc.) unless they're in the schedule
    fixture_prefixes = ("L-", "CF-", "S-", "E-", "EL-", "EM-", "SP-")
    # F- codes only kept if they appear in the schedule (avoids P-100 → F-100 misreads)
    filtered_counts = {}
    for code, count in fixture_counts.items():
        if code == "CEILING_FAN" or code.startswith(fixture_prefixes) or code in schedule:
            filtered_counts[code] = count
        elif code.startswith("F-") and code in schedule:
            filtered_counts[code] = count



    elapsed = time.monotonic() - started
    result = FixtureCountResult(
        page_name=image_path.stem,
        fixture_counts=filtered_counts,
        schedule_entries=schedule,
        occurrences=deduped,
        fan_count=fan_count,
        all_ocr_texts=all_raw_texts,
        metrics={
            "elapsed_s": round(elapsed, 3),
            "page_kind": "plan",
            "tile_count": tile_count,
            "blank_tiles_skipped": skipped_blank,
            "raw_ocr_texts": len(all_raw_texts),
            "raw_occurrences": raw_occurrence_count,
            "deduped_occurrences": len(deduped),
            "schedule_entries": len(schedule),
            "fans": fan_count,
            "fan_detection_method": fan_detection_method,
            **_calculate_reconciliation(
                FixtureCountResult(
                    page_name=image_path.stem,
                    fixture_counts=filtered_counts,
                    schedule_entries=schedule,
                    occurrences=deduped,
                    fan_count=fan_count,
                    all_ocr_texts=all_raw_texts,
                )
            ),
        },
    )

    total = sum(filtered_counts.values())
    logger.info("%s: %d fixture types, %d total count, %d fans in %.1fs",
                image_path.stem, len(filtered_counts), total, fan_count, elapsed)
    for code, count in sorted(filtered_counts.items()):
        desc = schedule.get(code, "")
        logger.info("  %s: %d%s", code, count, f" — {desc}" if desc else "")

    return result


# ── Legacy full-page OCR (fallback) ──────────────────────────────────────────


def count_fixtures_ocr(
    image_path: str | Path,
    lang: str = "en",
    search_radius: float = 300,
) -> FixtureCountResult:
    """OCR a full-resolution RCP page image (non-tiled fallback)."""
    image_path = Path(image_path)
    ocr = _get_ocr(lang)

    started = time.monotonic()
    result = _ocr_page_result(str(image_path), lang)
    if not result or not result[0]:
        return FixtureCountResult(page_name=image_path.stem, metrics={"elapsed_s": round(time.monotonic() - started, 3), "fan_detection_method": "none"})

    all_texts: list[tuple[list, str, float]] = []
    all_raw: list[str] = []
    for line in result[0]:
        text = line[1][0]
        conf = line[1][1]
        all_texts.append((line[0], text, conf))
        all_raw.append(text)

    img = cv2.imread(str(image_path))
    img_h, img_w = img.shape[:2]

    occurrences: list[FixtureOccurrence] = []
    schedule_entries: dict[str, str] = {}

    for points, text, conf in all_texts:
        bbox = _bbox_from_points(points)
        codes = FIXTURE_CODE_RE.findall(text)

        for code in codes:
            if not _is_fixture_code(code):
                continue

            is_schedule = (
                (bbox[0] + bbox[2]) / 2 > img_w * 0.72
                or (bbox[1] + bbox[3]) / 2 > img_h * 0.82
            )

            qty = _find_nearby_quantity_in_tile(bbox, all_texts, search_radius)

            if is_schedule:
                match = SCHEDULE_LINE_RE.search(text)
                if match:
                    schedule_entries[code] = match.group(2).strip()
                source = "schedule"
            else:
                source = "plan"

            occurrences.append(FixtureOccurrence(
                code=code, quantity=qty or 1, bbox=bbox,
                raw_text=text, confidence=conf, source=source,
            ))

    occurrences = _filter_partial_reads(occurrences)

    fixture_counts: dict[str, int] = defaultdict(int)
    for occ in occurrences:
        if occ.source == "plan":
            fixture_counts[occ.code] += occ.quantity
    for code in schedule_entries:
        if code not in fixture_counts:
            fixture_counts[code] = 0

    return FixtureCountResult(
        page_name=image_path.stem,
        fixture_counts=dict(fixture_counts),
        schedule_entries=schedule_entries,
        occurrences=occurrences,
        all_ocr_texts=all_raw,
        metrics={
            "elapsed_s": round(time.monotonic() - started, 3),
            "ocr_lines": len(all_texts),
            "occurrences": len(occurrences),
            "schedule_entries": len(schedule_entries),
            "fan_detection_method": "legacy",
            **_calculate_reconciliation(
                FixtureCountResult(
                    page_name=image_path.stem,
                    fixture_counts=dict(fixture_counts),
                    schedule_entries=schedule_entries,
                    occurrences=occurrences,
                    all_ocr_texts=all_raw,
                )
            ),
        },
    )


# ── Pipeline entry point ─────────────────────────────────────────────────────


def _extract_schedule_from_page(
    image_path: str | Path,
    lang: str = "en",
) -> dict[str, str]:
    """OCR an entire page and extract fixture schedule entries.

    Used for dedicated schedule pages in multi-page PDFs. Scans the full page
    for lines matching 'CODE  description' patterns, not just the margin regions.
    """
    image_path = Path(image_path)
    ocr = _get_ocr(lang)
    schedule: dict[str, str] = {}

    result = _ocr_page_result(str(image_path), lang)
    if not result or not result[0]:
        return schedule

    schedule.update(_parse_schedule_entries(result[0]))

    if schedule:
        logger.info("Schedule page extracted %d entries: %s", len(schedule), list(schedule.keys()))

    return schedule


SCHEDULE_PAGE_KEYWORDS = (
    "FIXTURE SCHEDULE",
    "LIGHTING SCHEDULE",
    "LIGHT FIXTURE SCHEDULE",
    "CEILING FIXTURE",
    "FIXTURE TYPE",
    "LAMP TYPE",
)
DETAIL_PAGE_KEYWORDS = (
    "DETAIL",
    "SECTION",
    "ELEVATION",
    "TYPICAL",
    "SCALE",
)
IRRELEVANT_PAGE_KEYWORDS = (
    "COVER SHEET",
    "TITLE SHEET",
    "INDEX",
    "TRANSMITTAL",
    "NOT FOR CONSTRUCTION",
)


@lru_cache(maxsize=256)
def _is_nearly_blank_image(image_path: str | Path, white_threshold: float = 0.995) -> bool:
    """Fast pre-check used to skip obviously blank pages before OCR."""
    img = cv2.imread(str(image_path))
    if img is None:
        return False
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    # Downsample to keep the pre-check cheap on high-res pages.
    if gray.shape[0] > 1000 or gray.shape[1] > 1000:
        gray = cv2.resize(gray, (max(1, gray.shape[1] // 4), max(1, gray.shape[0] // 4)))
    white_ratio = float(np.mean(gray > 250))
    return white_ratio >= white_threshold


def _classify_page_text(all_text: str, fixture_codes: list[str] | None = None) -> tuple[str, dict[str, int | str]]:
    """Classify a page by OCR text before expensive counting work.

    Returns one of: plan, schedule, detail, irrelevant.
    """
    text = all_text.upper()
    codes = fixture_codes if fixture_codes is not None else FIXTURE_CODE_RE.findall(text)
    code_count = len(codes)
    keyword_hits = sum(1 for kw in SCHEDULE_PAGE_KEYWORDS if kw in text)
    detail_hits = sum(1 for kw in DETAIL_PAGE_KEYWORDS if kw in text)
    irrelevant_hits = sum(1 for kw in IRRELEVANT_PAGE_KEYWORDS if kw in text)
    word_count = len(re.findall(r"[A-Z0-9]+", text))

    if keyword_hits > 0 or code_count >= 10:
        return "schedule", {
            "reason": "schedule_keyword" if keyword_hits else "fixture_code_density",
            "fixture_code_count": code_count,
            "keyword_hits": keyword_hits,
            "detail_hits": detail_hits,
            "irrelevant_hits": irrelevant_hits,
        }

    if irrelevant_hits > 0 and code_count == 0:
        return "irrelevant", {
            "reason": "irrelevant_keyword",
            "fixture_code_count": code_count,
            "keyword_hits": keyword_hits,
            "detail_hits": detail_hits,
            "irrelevant_hits": irrelevant_hits,
        }

    if detail_hits > 0 and code_count == 0:
        return "detail", {
            "reason": "detail_keyword",
            "fixture_code_count": code_count,
            "keyword_hits": keyword_hits,
            "detail_hits": detail_hits,
            "irrelevant_hits": irrelevant_hits,
        }

    if code_count == 0 and word_count < 8:
        return "irrelevant", {
            "reason": "low_text_density",
            "fixture_code_count": code_count,
            "keyword_hits": keyword_hits,
            "detail_hits": detail_hits,
            "irrelevant_hits": irrelevant_hits,
        }

    return "plan", {
        "reason": "default",
        "fixture_code_count": code_count,
        "keyword_hits": keyword_hits,
        "detail_hits": detail_hits,
        "irrelevant_hits": irrelevant_hits,
    }



def _classify_page_ocr(result: list) -> tuple[str, dict[str, int | str]]:
    if not result or not result[0]:
        return "irrelevant", {"reason": "empty_ocr", "fixture_code_count": 0, "keyword_hits": 0, "detail_hits": 0, "irrelevant_hits": 0}
    all_text = " ".join(line[1][0] for line in result[0])
    return _classify_page_text(all_text)



def _is_schedule_page(image_path: str | Path, lang: str = "en") -> bool:
    """Backward-compatible schedule-page check."""
    result = _ocr_page_result(str(Path(image_path)), lang)
    page_kind, _meta = _classify_page_ocr(result)
    return page_kind == "schedule"


def count_fixtures_from_pdf(
    pdf_path: str | Path,
    pages_dir: str | Path | None = None,
    dpi: int = 300,
    lang: str = "en",
    use_tiling: bool = True,
    detect_fans: bool = True,
    **kwargs,
) -> list[FixtureCountResult]:
    """Full pipeline: PDF → PNG → OCR → fixture counts.

    Renders ALL pages of the PDF. Plan pages get full tiled OCR + fan detection.
    Schedule pages are parsed for fixture code/description tables and merged
    into plan results.

    Set *use_tiling=True* (default) for spatial-dedup tile-based counting.
    Set *detect_fans=True* to include template-based ceiling fan detection.
    """
    pdf_path = Path(pdf_path)

    if pages_dir is not None:
        pages_dir = Path(pages_dir)
        image_paths = sorted(pages_dir.glob("*.png"))
    else:
        from rcp_detector.pdf.converter import pdf_to_pngs
        pages_dir = Path("output") / pdf_path.stem / "pages"
        image_paths = pdf_to_pngs(pdf_path, pages_dir, dpi=dpi)

    total_pages = len(image_paths)
    started = time.monotonic()
    logger.info("Starting OCR pass for %d pages from %s", total_pages, pdf_path.name)

    # Two-pass: classify pages first, then OCR only the pages we need to count.
    schedule_entries: dict[str, str] = {}
    plan_pages: list[tuple[int, Path, str]] = []
    page_type_counts = defaultdict(int)

    for page_num, img_path in enumerate(image_paths, start=1):
        if total_pages > 1 and _is_nearly_blank_image(img_path):
            page_kind = "irrelevant"
            page_meta = {"reason": "blank_image", "fixture_code_count": 0, "keyword_hits": 0, "detail_hits": 0, "irrelevant_hits": 0}
        else:
            page_ocr = _ocr_page_result(str(img_path), lang)
            page_kind, page_meta = _classify_page_ocr(page_ocr)

        page_type_counts[page_kind] += 1

        logger.info(
            "Classified page %d/%d as %s (%s): %s",
            page_num,
            total_pages,
            page_kind,
            page_meta.get("reason", "n/a"),
            img_path.name,
        )

        if page_kind == "schedule":
            page_schedule = _extract_schedule_from_page(img_path, lang)
            if page_schedule:
                logger.info(
                    "Detected schedule page %d/%d: %s (%d entries)",
                    page_num,
                    total_pages,
                    img_path.name,
                    len(page_schedule),
                )
                schedule_entries.update(page_schedule)
            continue

        if total_pages > 1 and page_kind in {"detail", "irrelevant"}:
            logger.info("Skipping non-target page %d/%d: %s", page_num, total_pages, img_path.name)
            continue

        plan_pages.append((page_num, img_path, page_kind))

    if schedule_entries:
        logger.info("Multi-page schedule: %d fixture descriptions extracted", len(schedule_entries))
    logger.info(
        "Page classification summary: plan=%d schedule=%d detail=%d irrelevant=%d",
        page_type_counts.get("plan", 0),
        page_type_counts.get("schedule", 0),
        page_type_counts.get("detail", 0),
        page_type_counts.get("irrelevant", 0),
    )

    results = []
    for page_num, img_path, page_kind in plan_pages:
        page_started = time.monotonic()
        logger.info("Page %d/%d start: %s", page_num, total_pages, img_path.name)
        if use_tiling:
            result = count_fixtures_tiled(img_path, detect_fans=detect_fans, lang=lang, dpi=dpi, **kwargs)
        else:
            result = count_fixtures_ocr(img_path, lang=lang)

        if schedule_entries:
            result.schedule_entries.update(schedule_entries)
            for code in schedule_entries:
                if code not in result.fixture_counts:
                    fixture_prefixes = ("L-", "CF-", "F-", "S-", "E-", "EL-", "EM-", "SP-")
                    if code.startswith(fixture_prefixes):
                        result.fixture_counts[code] = 0

        result.metrics.update({
            "page_num": page_num,
            "page_total": total_pages,
            "page_kind": page_kind,
            "page_elapsed_s": round(time.monotonic() - page_started, 3),
            **_calculate_reconciliation(result),
        })
        results.append(result)
        logger.info("Page %d/%d complete in %.1fs: %d types, %d total, %d fans", page_num, total_pages, result.metrics["page_elapsed_s"], len(result.fixture_counts), sum(result.fixture_counts.values()), result.fan_count)

    logger.info("Completed OCR pass for %d pages in %.1fs", len(results), time.monotonic() - started)
    return results


def _calculate_reconciliation(result: FixtureCountResult) -> dict[str, int | float]:
    """Summarize schedule-vs-plan agreement and OCR confidence for a page."""
    schedule_codes = set(result.schedule_entries or {})
    fixture_counts = result.fixture_counts or {}
    occurrences = result.occurrences or []

    zero_schedule_codes = sum(1 for code in schedule_codes if int(fixture_counts.get(code, 0) or 0) == 0)
    unscheduled_codes = sum(1 for code in fixture_counts if code not in schedule_codes and code != "CEILING_FAN")

    confidences = [float(occ.confidence) for occ in occurrences if getattr(occ, "confidence", 0.0) is not None]
    low_conf_threshold = 0.60
    low_conf_occurrences = sum(1 for conf in confidences if conf < low_conf_threshold)
    avg_confidence = round(sum(confidences) / len(confidences), 3) if confidences else 0.0
    min_confidence = round(min(confidences), 3) if confidences else 0.0

    return {
        "schedule_codes": len(schedule_codes),
        "zero_schedule_codes": zero_schedule_codes,
        "unscheduled_codes": unscheduled_codes,
        "low_confidence_occurrences": low_conf_occurrences,
        "avg_confidence": avg_confidence,
        "min_confidence": min_confidence,
    }


# ── Markdown output ──────────────────────────────────────────────────────────


def _summarize_results(results: list[FixtureCountResult]) -> dict[str, int | float]:
    summary = {
        "pages": len(results),
        "elapsed_s": 0.0,
        "fixture_types": 0,
        "fixture_total": 0,
        "raw_ocr_texts": 0,
        "raw_occurrences": 0,
        "deduped_occurrences": 0,
        "tiles": 0,
        "blank_tiles_skipped": 0,
        "schedule_entries": 0,
        "zero_schedule_codes": 0,
        "unscheduled_codes": 0,
        "low_confidence_occurrences": 0,
        "confidence_sum": 0.0,
        "confidence_count": 0,
        "min_confidence": None,
        "fans": 0,
    }

    for r in results:
        metrics = r.metrics or {}
        summary["elapsed_s"] += float(metrics.get("elapsed_s", metrics.get("page_elapsed_s", 0.0)) or 0.0)
        summary["fixture_types"] += len(r.fixture_counts or {})
        summary["fixture_total"] += sum(int(v) for v in (r.fixture_counts or {}).values())
        summary["raw_ocr_texts"] += int(metrics.get("raw_ocr_texts", metrics.get("ocr_lines", 0)) or 0)
        summary["raw_occurrences"] += int(metrics.get("raw_occurrences", metrics.get("occurrences", 0)) or 0)
        summary["deduped_occurrences"] += int(metrics.get("deduped_occurrences", len(r.occurrences)) or 0)
        summary["tiles"] += int(metrics.get("tile_count", 0) or 0)
        summary["blank_tiles_skipped"] += int(metrics.get("blank_tiles_skipped", 0) or 0)
        summary["schedule_entries"] += int(metrics.get("schedule_entries", len(r.schedule_entries)) or 0)
        summary["zero_schedule_codes"] += int(metrics.get("zero_schedule_codes", 0) or 0)
        summary["unscheduled_codes"] += int(metrics.get("unscheduled_codes", 0) or 0)
        summary["low_confidence_occurrences"] += int(metrics.get("low_confidence_occurrences", 0) or 0)
        occ_confidences = [float(getattr(occ, "confidence", 0.0) or 0.0) for occ in (r.occurrences or [])]
        summary["confidence_sum"] += sum(occ_confidences)
        summary["confidence_count"] += len(occ_confidences)
        if occ_confidences:
            page_min_conf = min(occ_confidences)
            summary["min_confidence"] = page_min_conf if summary["min_confidence"] is None else min(summary["min_confidence"], page_min_conf)
        summary["fans"] += int(metrics.get("fans", r.fan_count) or 0)

    summary["elapsed_s"] = round(summary["elapsed_s"], 3)
    if summary["confidence_count"]:
        summary["avg_confidence"] = round(summary["confidence_sum"] / summary["confidence_count"], 3)
        summary["min_confidence"] = round(summary["min_confidence"], 3) if summary["min_confidence"] is not None else 0.0
    else:
        summary["avg_confidence"] = 0.0
        summary["min_confidence"] = 0.0
    return summary


def format_results_markdown(results: list[FixtureCountResult]) -> str:
    """Format OCR fixture count results as markdown."""
    lines = ["# OCR Fixture Count Results", ""]

    if results:
        summary = _summarize_results(results)
        lines.append("## Run Summary")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|------:|")
        lines.append(f"| Pages | {summary['pages']} |")
        lines.append(f"| Total elapsed | {summary['elapsed_s']} s |")
        lines.append(f"| Fixture types | {summary['fixture_types']} |")
        lines.append(f"| Fixture total | {summary['fixture_total']} |")
        lines.append(f"| OCR texts | {summary['raw_ocr_texts']} |")
        lines.append(f"| Raw hits | {summary['raw_occurrences']} |")
        lines.append(f"| Deduped hits | {summary['deduped_occurrences']} |")
        lines.append(f"| Tiles scanned | {summary['tiles']} |")
        lines.append(f"| Blank tiles skipped | {summary['blank_tiles_skipped']} |")
        lines.append(f"| Schedule entries | {summary['schedule_entries']} |")
        lines.append(f"| Schedule zero-counts | {summary['zero_schedule_codes']} |")
        lines.append(f"| Unscheduled codes | {summary['unscheduled_codes']} |")
        lines.append(f"| Low-confidence hits | {summary['low_confidence_occurrences']} |")
        lines.append(f"| Avg confidence | {summary['avg_confidence']} |")
        lines.append(f"| Min confidence | {summary['min_confidence']} |")
        lines.append(f"| Fans | {summary['fans']} |")
        lines.append("")

    for r in results:
        lines.append(f"### {r.page_name}")
        lines.append("")

        if r.metrics:
            lines.append("**Run Metrics:**")
            lines.append("")
            metric_bits = []
            for key, label in (
                ("elapsed_s", "Elapsed"),
                ("page_kind", "Page Kind"),
                ("page_elapsed_s", "Page Elapsed"),
                ("raw_ocr_texts", "OCR Texts"),
                ("raw_occurrences", "Raw Hits"),
                ("deduped_occurrences", "Deduped Hits"),
                ("tile_count", "Tiles"),
                ("blank_tiles_skipped", "Blank Tiles Skipped"),
                ("schedule_entries", "Schedule Entries"),
                ("zero_schedule_codes", "Schedule Zero-Counts"),
                ("unscheduled_codes", "Unscheduled Codes"),
                ("low_confidence_occurrences", "Low-Confidence Hits"),
                ("avg_confidence", "Avg Confidence"),
                ("min_confidence", "Min Confidence"),
                ("fans", "Fans"),
                ("fan_detection_method", "Fan Detection Method"),
            ):
                if key in r.metrics:
                    metric_bits.append(f"{label}: {r.metrics[key]}")
            if metric_bits:
                lines.append("- " + " | ".join(metric_bits))
                lines.append("")

        if not r.fixture_counts:
            lines.append("*No fixture codes detected.*")
            lines.append("")
            continue

        reconciliation = _calculate_reconciliation(r)
        if reconciliation["zero_schedule_codes"] or reconciliation["unscheduled_codes"] or reconciliation["low_confidence_occurrences"]:
            lines.append("**Reconciliation:**")
            lines.append("")
            lines.append("| Metric | Value |")
            lines.append("|--------|------:|")
            lines.append(f"| Schedule codes | {reconciliation['schedule_codes']} |")
            lines.append(f"| Schedule zero-counts | {reconciliation['zero_schedule_codes']} |")
            lines.append(f"| Unscheduled codes | {reconciliation['unscheduled_codes']} |")
            lines.append(f"| Low-confidence hits | {reconciliation['low_confidence_occurrences']} |")
            lines.append(f"| Avg confidence | {reconciliation['avg_confidence']} |")
            lines.append(f"| Min confidence | {reconciliation['min_confidence']} |")
            lines.append("")

        lights = {k: v for k, v in r.fixture_counts.items() if k.startswith("L-")}
        fans = {k: v for k, v in r.fixture_counts.items()
                if k.startswith(("CF-", "F-")) or k == "CEILING_FAN"}
        other = {k: v for k, v in r.fixture_counts.items() if k not in lights and k not in fans}

        if lights:
            lines.append("**Light Fixtures:**")
            lines.append("")
            lines.append("| Fixture Code | Count | Description |")
            lines.append("|-------------|------:|-------------|")
            for code, count in sorted(lights.items()):
                desc = r.schedule_entries.get(code, "")
                lines.append(f"| {code} | {count} | {desc} |")
            lines.append(f"| **Subtotal** | **{sum(lights.values())}** | |")
            lines.append("")

        if fans:
            lines.append("**Ceiling Fans:**")
            lines.append("")
            lines.append("| Source | Count |")
            lines.append("|--------|------:|")
            for code, count in sorted(fans.items()):
                label = "Visual detection (template)" if code == "CEILING_FAN" else code
                lines.append(f"| {label} | {count} |")
            lines.append(f"| **Subtotal** | **{sum(fans.values())}** | |")
            lines.append("")

        if other:
            lines.append("**Other Fixtures:**")
            lines.append("")
            lines.append("| Fixture Code | Count | Description |")
            lines.append("|-------------|------:|-------------|")
            for code, count in sorted(other.items()):
                desc = r.schedule_entries.get(code, "")
                lines.append(f"| {code} | {count} | {desc} |")
            lines.append(f"| **Subtotal** | **{sum(other.values())}** | |")
            lines.append("")

        total = sum(r.fixture_counts.values())
        lines.append(f"**Page Total: {total} fixtures**")
        lines.append("")

    return "\n".join(lines)
