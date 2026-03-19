"""Count label distribution across a YOLO dataset.

Refactored from count_labels_script_with_Total.py — no hardcoded paths.
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def count_labels(
    labels_dir: str | Path,
    classes_file: str | Path | None = None,
    class_names: list[str] | None = None,
) -> dict[str, int]:
    """Count bounding boxes per class in a directory of YOLO label files.

    Provide either *classes_file* (path to newline-delimited class names) or
    *class_names* directly. Returns a dict of class_name → count.
    """
    labels_dir = Path(labels_dir)

    if class_names is None:
        if classes_file is None:
            raise ValueError("Provide either classes_file or class_names")
        with open(classes_file) as f:
            class_names = [line.strip() for line in f if line.strip()]

    counts = {name: 0 for name in class_names}

    for label_file in labels_dir.glob("*.txt"):
        with open(label_file) as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                cls_idx = int(parts[0])
                if 0 <= cls_idx < len(class_names):
                    counts[class_names[cls_idx]] += 1

    total = sum(counts.values())
    logger.info("Total bounding boxes: %d across %d classes", total, len(class_names))
    return counts


def print_label_stats(counts: dict[str, int]) -> None:
    """Pretty-print label distribution."""
    total = sum(counts.values())
    for name, count in counts.items():
        pct = (count / total * 100) if total > 0 else 0
        print(f"  {name}: {count} ({pct:.1f}%)")
    print(f"\n  Total: {total}")
