"""OCR-first fixture counter — extract fixture types and counts directly from RCP text.

Scans full-resolution RCP page images with OCR, then uses regex to find:
- Fixture codes (L-200, L-100A, CF-1, etc.)
- Quantity callouts (QTY 3, (2), etc.)
- Fixture schedule tables (legend area)
"""

import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

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

# Fixture codes: L-200, L-100A, CF-1, S-101, EL-1, etc.
FIXTURE_CODE_RE = re.compile(
    r"\b([A-Z]{1,3}-\d{1,4}[A-Z]?)\b"
)

# Quantity patterns: QTY 3, QTY. 4, (2), (3), QTY:5
QTY_EXPLICIT_RE = re.compile(
    r"QTY\.?\s*:?\s*(\d+)", re.IGNORECASE
)
QTY_PAREN_RE = re.compile(
    r"\((\d{1,3})\)"
)

# Schedule line: "L-200  RECESSED DOWNLIGHT  ..."  (fixture code at start of a text block)
SCHEDULE_LINE_RE = re.compile(
    r"\b([A-Z]{1,3}-\d{1,4}[A-Z]?)\b\s+(.+)", re.IGNORECASE
)


@dataclass
class FixtureOccurrence:
    """A single occurrence of a fixture code found in the drawing."""
    code: str
    quantity: int  # from nearby QTY tag, or 1 if standalone
    bbox: list  # [x1, y1, x2, y2] pixel coordinates
    raw_text: str  # the full OCR text line
    source: str = "plan"  # "plan" or "schedule"


@dataclass
class FixtureCountResult:
    """Aggregated results for one page."""
    page_name: str
    fixture_counts: dict = field(default_factory=dict)  # code → total count
    schedule_entries: dict = field(default_factory=dict)  # code → description from schedule
    occurrences: list = field(default_factory=list)  # all FixtureOccurrence objects
    all_ocr_texts: list = field(default_factory=list)  # raw OCR output for debugging


def _bbox_from_points(points: list) -> list[float]:
    """Convert PaddleOCR polygon points to [x1, y1, x2, y2]."""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return [min(xs), min(ys), max(xs), max(ys)]


def _find_nearby_quantity(
    target_bbox: list[float],
    all_texts: list[tuple[list, str]],
    search_radius: float = 300,
) -> int | None:
    """Search nearby OCR text boxes for a quantity callout."""
    tx1, ty1, tx2, ty2 = target_bbox
    tcx = (tx1 + tx2) / 2
    tcy = (ty1 + ty2) / 2

    for points, text in all_texts:
        bbox = _bbox_from_points(points)
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2

        dist = ((tcx - cx) ** 2 + (tcy - cy) ** 2) ** 0.5
        if dist > search_radius:
            continue

        # Check for QTY pattern
        qty_match = QTY_EXPLICIT_RE.search(text)
        if qty_match:
            return int(qty_match.group(1))

        # Check for parenthesized number
        paren_match = QTY_PAREN_RE.search(text)
        if paren_match:
            val = int(paren_match.group(1))
            if 1 <= val <= 99:  # sanity check
                return val

    return None


def _is_in_schedule_region(
    bbox: list[float],
    img_width: int,
    img_height: int,
) -> bool:
    """Heuristic: schedule/legend is typically in the right 30% or bottom 20% of the page."""
    cx = (bbox[0] + bbox[2]) / 2
    cy = (bbox[1] + bbox[3]) / 2
    # Right edge region
    if cx > img_width * 0.72:
        return True
    # Bottom region
    if cy > img_height * 0.82:
        return True
    return False


def count_fixtures_ocr(
    image_path: str | Path,
    lang: str = "en",
    search_radius: float = 300,
) -> FixtureCountResult:
    """OCR a full-resolution RCP page image and extract fixture counts.

    Returns a FixtureCountResult with fixture codes, quantities, and locations.
    """
    image_path = Path(image_path)
    ocr = _get_ocr(lang)

    logger.info("Running OCR on %s ...", image_path.name)
    result = ocr.ocr(str(image_path), cls=True)

    if not result or not result[0]:
        logger.warning("No OCR text found in %s", image_path.name)
        return FixtureCountResult(page_name=image_path.stem)

    # Collect all text with positions
    all_texts: list[tuple[list, str]] = []  # (polygon_points, text)
    all_raw: list[str] = []

    for line in result[0]:
        points = line[0]  # [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
        text = line[1][0]  # text string
        confidence = line[1][1]
        all_texts.append((points, text))
        all_raw.append(text)

    logger.info("OCR found %d text regions", len(all_texts))

    # Get image dimensions for schedule region detection
    import cv2
    img = cv2.imread(str(image_path))
    img_h, img_w = img.shape[:2]

    # Pass 1: Find fixture codes and nearby quantities
    occurrences: list[FixtureOccurrence] = []
    schedule_entries: dict[str, str] = {}

    for points, text in all_texts:
        bbox = _bbox_from_points(points)

        # Look for fixture codes in this text
        code_matches = FIXTURE_CODE_RE.findall(text)

        for code in code_matches:
            # Skip codes that are clearly not fixtures (page numbers, dates, etc.)
            if code in ("ID-2", "ID-8", "A-1", "A-2", "A-3"):
                continue
            # Skip if it looks like a sheet number (single digit after dash)
            if re.match(r"^[A-Z]-\d$", code) and code[0] not in ("L", "C", "S", "E"):
                continue

            is_schedule = _is_in_schedule_region(bbox, img_w, img_h)

            # Try to find a quantity nearby
            qty = _find_nearby_quantity(bbox, all_texts, search_radius)

            if is_schedule:
                # In the schedule area — extract description
                sched_match = SCHEDULE_LINE_RE.search(text)
                desc = sched_match.group(2).strip() if sched_match else text
                schedule_entries[code] = desc
                source = "schedule"
            else:
                source = "plan"

            occurrences.append(FixtureOccurrence(
                code=code,
                quantity=qty if qty else 1,
                bbox=bbox,
                raw_text=text,
                source=source,
            ))

    # Aggregate counts
    # Plan occurrences: each is an instance (possibly with QTY multiplier)
    # Schedule occurrences: these describe the fixture, not count it
    fixture_counts: dict[str, int] = defaultdict(int)

    for occ in occurrences:
        if occ.source == "plan":
            fixture_counts[occ.code] += occ.quantity
        # Schedule entries don't add to count — they're descriptions

    # For codes only found in schedule (no plan instances), note them with 0
    for code in schedule_entries:
        if code not in fixture_counts:
            fixture_counts[code] = 0

    result = FixtureCountResult(
        page_name=image_path.stem,
        fixture_counts=dict(fixture_counts),
        schedule_entries=schedule_entries,
        occurrences=occurrences,
        all_ocr_texts=all_raw,
    )

    # Log summary
    total = sum(fixture_counts.values())
    logger.info(
        "%s: found %d fixture codes, %d total count, %d schedule entries",
        image_path.stem, len(fixture_counts), total, len(schedule_entries),
    )
    for code, count in sorted(fixture_counts.items()):
        desc = schedule_entries.get(code, "")
        logger.info("  %s: %d%s", code, count, f" — {desc}" if desc else "")

    return result


def count_fixtures_from_pdf(
    pdf_path: str | Path,
    pages_dir: str | Path | None = None,
    dpi: int = 300,
    lang: str = "en",
    search_radius: float = 300,
) -> list[FixtureCountResult]:
    """Full pipeline: PDF → PNG → OCR → fixture counts.

    If *pages_dir* contains pre-rendered PNGs, uses those. Otherwise renders the PDF.
    """
    pdf_path = Path(pdf_path)
    results = []

    # Get or render page images
    if pages_dir is not None:
        pages_dir = Path(pages_dir)
        image_paths = sorted(pages_dir.glob("*.png"))
    else:
        from rcp_detector.pdf.converter import pdf_to_pngs
        pages_dir = Path("output") / pdf_path.stem / "pages"
        image_paths = pdf_to_pngs(pdf_path, pages_dir, dpi=dpi)

    for img_path in image_paths:
        result = count_fixtures_ocr(img_path, lang=lang, search_radius=search_radius)
        results.append(result)

    return results


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

        # Separate lights, fans, and other
        lights = {k: v for k, v in r.fixture_counts.items() if k.startswith("L-")}
        fans = {k: v for k, v in r.fixture_counts.items() if k.startswith(("CF-", "F-"))}
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
            lines.append("| Fixture Code | Count | Description |")
            lines.append("|-------------|------:|-------------|")
            for code, count in sorted(fans.items()):
                desc = r.schedule_entries.get(code, "")
                lines.append(f"| {code} | {count} | {desc} |")
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
