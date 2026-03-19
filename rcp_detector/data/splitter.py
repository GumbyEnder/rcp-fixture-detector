"""Split a YOLO dataset into train/valid/test splits.

Refactored from create_data_split.py — no hardcoded paths.
"""

import logging
import random
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)


def split_dataset(
    images_dir: str | Path,
    labels_dir: str | Path,
    output_dir: str | Path,
    train_ratio: float = 0.7,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> dict[str, int]:
    """Split images + labels into train/valid/test directories.

    Returns a dict with counts per split.
    """
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    output_dir = Path(output_dir)

    splits = {"train": train_ratio, "test": test_ratio, "valid": 1.0 - train_ratio - test_ratio}

    # Create split directories
    for split_name in splits:
        (output_dir / split_name / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / split_name / "labels").mkdir(parents=True, exist_ok=True)

    # Find paired image+label files
    image_files = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    pairs = []
    for img in image_files:
        lbl = labels_dir / (img.stem + ".txt")
        if lbl.exists():
            pairs.append((img, lbl))
        else:
            logger.warning("No label for %s — skipping", img.name)

    random.seed(seed)
    random.shuffle(pairs)

    n = len(pairs)
    n_train = int(n * train_ratio)
    n_test = int(n * test_ratio)

    assignments = {
        "train": pairs[:n_train],
        "test": pairs[n_train:n_train + n_test],
        "valid": pairs[n_train + n_test:],
    }

    counts = {}
    for split_name, split_pairs in assignments.items():
        for img_path, lbl_path in split_pairs:
            shutil.copy2(img_path, output_dir / split_name / "images" / img_path.name)
            shutil.copy2(lbl_path, output_dir / split_name / "labels" / lbl_path.name)
        counts[split_name] = len(split_pairs)
        logger.info("%s: %d samples", split_name, len(split_pairs))

    return counts
