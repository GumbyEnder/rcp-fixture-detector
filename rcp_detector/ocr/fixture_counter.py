"""OCR-first fixture counter — extract fixture types and counts directly from RCP text.

Improvements over v1:
1. Tile-based OCR with spatial deduplication (fixes overlap inflation)
2. Ceiling fan detection via Hough circles (Phase 2)
3. Schedule table parsing from legend region
4. Partial read filtering (L-10 → L-100)
5. Configurable DPI (already wired in CLI)
"""

import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Lazy-init to avoid import cost
_ocr_instance = None


def _get_ocr(lang: str = "en"):
    global _ocr_instance
    if _ocr_instance is None:
        from paddleocr import PaddleOCR
        _ocr_instance = PaddleOCR(use_angle_cls=True, lang=lang, show_log=False)
    return _ocr_instance


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

    E.g., if both L-10 and L-100 exist, drop L-10.
    """
    all_codes = {occ.code for occ in occurrences}

    def is_partial(code: str) -> bool:
        for other in all_codes:
            if other != code and other.startswith(code) and len(other) > len(code):
                return True
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

        result = ocr.ocr(crop, cls=True)
        if not result or not result[0]:
            continue

        # Collect all text in this region and look for schedule lines
        for line in result[0]:
            text = line[1][0]
            match = SCHEDULE_LINE_RE.search(text)
            if match:
                code = match.group(1)
                desc = match.group(2).strip()
                if _is_fixture_code(code) and len(desc) > 2:
                    schedule[code] = desc

    if schedule:
        logger.info("Schedule entries found: %s", list(schedule.keys()))

    return schedule


# ── Core: Tile-based OCR ─────────────────────────────────────────────────────


def _find_nearby_quantity_in_tile(
    target_bbox: list[float],
    all_texts: list[tuple[list, str, float]],
    search_radius: float = 200,
) -> int | None:
    """Search nearby OCR text boxes within a single tile for a QTY callout."""
    tcx = (target_bbox[0] + target_bbox[2]) / 2
    tcy = (target_bbox[1] + target_bbox[3]) / 2

    for points, text, _conf in all_texts:
        bbox = _bbox_from_points(points)
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        dist = ((tcx - cx) ** 2 + (tcy - cy) ** 2) ** 0.5
        if dist > search_radius:
            continue

        qty_match = QTY_EXPLICIT_RE.search(text)
        if qty_match:
            return int(qty_match.group(1))

        paren_match = QTY_PAREN_RE.search(text)
        if paren_match:
            val = int(paren_match.group(1))
            if 1 <= val <= 99:
                return val

    return None


def count_fixtures_tiled(
    image_path: str | Path,
    patch_size: int = 640,
    overlap_factor: float = 0.5,
    lang: str = "en",
    search_radius: float = 200,
    dedup_iou: float = 0.3,
    dedup_dist: float = 40.0,
    detect_fans: bool = False,  # experimental — high FP rate on architectural drawings
    fan_min_radius: int = 8,
    fan_max_radius: int = 25,
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

    step = int(patch_size * (1 - overlap_factor))
    all_raw_texts: list[str] = []
    all_occurrences: list[FixtureOccurrence] = []
    fan_detections: list[dict] = []

    # ── Step 1: Extract schedule from legend area first ──
    schedule = _extract_schedule(img, lang)

    # ── Step 2: Tile and OCR ──
    tile_count = 0
    y = 0
    while y + patch_size <= img_h:
        x = 0
        while x + patch_size <= img_w:
            tile = img[y:y + patch_size, x:x + patch_size]

            # OCR this tile
            result = ocr.ocr(tile, cls=True)
            if result and result[0]:
                tile_texts: list[tuple[list, str, float]] = []
                for line in result[0]:
                    points = line[0]
                    text = line[1][0]
                    conf = line[1][1]
                    if conf > 0.5:
                        tile_texts.append((points, text, conf))
                        all_raw_texts.append(text)

                # Extract fixture codes from this tile
                for points, text, conf in tile_texts:
                    local_bbox = _bbox_from_points(points)
                    codes = FIXTURE_CODE_RE.findall(text)

                    for code in codes:
                        if not _is_fixture_code(code):
                            continue

                        # Map local bbox to page coordinates
                        page_bbox = [
                            local_bbox[0] + x,
                            local_bbox[1] + y,
                            local_bbox[2] + x,
                            local_bbox[3] + y,
                        ]

                        qty = _find_nearby_quantity_in_tile(local_bbox, tile_texts, search_radius)

                        all_occurrences.append(FixtureOccurrence(
                            code=code,
                            quantity=qty if qty else 1,
                            bbox=page_bbox,
                            raw_text=text,
                            confidence=conf,
                            source="plan",
                        ))

            # ── Step 3: Fan detection on this tile ──
            if detect_fans:
                from rcp_detector.detection.fan_detector import detect_ceiling_fans
                fans = detect_ceiling_fans(
                    tile,
                    min_radius=fan_min_radius,
                    max_radius=fan_max_radius,
                    min_blade_lines=4,
                    hough_param2=30,
                )
                for fan in fans:
                    fan["center_x"] += x  # map to page coords
                    fan["center_y"] += y
                    fan_detections.append(fan)

            tile_count += 1
            x += step
        y += step

    logger.info("OCR scanned %d tiles, found %d raw occurrences", tile_count, len(all_occurrences))

    # ── Step 4: Filter partial reads ──
    all_occurrences = _filter_partial_reads(all_occurrences)

    # ── Step 5: Spatial deduplication ──
    deduped = _spatial_dedup(all_occurrences, iou_threshold=dedup_iou, center_dist_threshold=dedup_dist)
    logger.info("After dedup: %d unique occurrences (was %d)", len(deduped), len(all_occurrences))

    # ── Step 6: Fan dedup ──
    fan_count = 0
    if fan_detections:
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
    fixture_prefixes = ("L-", "CF-", "F-", "S-", "E-", "EL-", "EM-", "SP-")
    filtered_counts = {}
    for code, count in fixture_counts.items():
        if code == "CEILING_FAN" or code.startswith(fixture_prefixes) or code in schedule:
            filtered_counts[code] = count

    result = FixtureCountResult(
        page_name=image_path.stem,
        fixture_counts=filtered_counts,
        schedule_entries=schedule,
        occurrences=deduped,
        fan_count=fan_count,
        all_ocr_texts=all_raw_texts,
    )

    total = sum(filtered_counts.values())
    logger.info("%s: %d fixture types, %d total count, %d fans",
                image_path.stem, len(filtered_counts), total, fan_count)
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

    result = ocr.ocr(str(image_path), cls=True)
    if not result or not result[0]:
        return FixtureCountResult(page_name=image_path.stem)

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
    )


# ── Pipeline entry point ─────────────────────────────────────────────────────


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

    Set *use_tiling=True* (default) for spatial-dedup tile-based counting.
    Set *detect_fans=True* to include Hough circle ceiling fan detection.
    """
    pdf_path = Path(pdf_path)

    if pages_dir is not None:
        pages_dir = Path(pages_dir)
        image_paths = sorted(pages_dir.glob("*.png"))
    else:
        from rcp_detector.pdf.converter import pdf_to_pngs
        pages_dir = Path("output") / pdf_path.stem / "pages"
        image_paths = pdf_to_pngs(pdf_path, pages_dir, dpi=dpi)

    results = []
    for img_path in image_paths:
        if use_tiling:
            result = count_fixtures_tiled(img_path, detect_fans=detect_fans, lang=lang, **kwargs)
        else:
            result = count_fixtures_ocr(img_path, lang=lang)
        results.append(result)

    return results


# ── Markdown output ──────────────────────────────────────────────────────────


def format_results_markdown(results: list[FixtureCountResult]) -> str:
    """Format OCR fixture count results as markdown."""
    lines = []

    for r in results:
        lines.append(f"### {r.page_name}")
        lines.append("")

        if not r.fixture_counts:
            lines.append("*No fixture codes detected.*")
            lines.append("")
            continue

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
                label = "Visual detection (Hough)" if code == "CEILING_FAN" else code
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
