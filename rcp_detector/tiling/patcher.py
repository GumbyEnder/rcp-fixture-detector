"""Tile large images into overlapping patches for inference or annotation.

Refactored from create_Overlaping_patched.py — config-driven, no hardcoded paths.
Stores tiling metadata alongside patches so reconstruction can recover the original resolution.
"""

import json
import logging
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class TilingMetadata(NamedTuple):
    image_name: str
    image_height: int
    image_width: int
    patch_height: int
    patch_width: int
    overlap_factor: float
    num_patches: int


def _xywh_to_xyxy(lines: list[str], img_h: int, img_w: int) -> list[list[int]]:
    """Convert YOLO normalized xywh labels to absolute xyxy."""
    labels = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls = int(parts[0])
        x, y, w, h = map(float, parts[1:5])
        x *= img_w
        y *= img_h
        w *= img_w
        h *= img_h
        left = int(x - w / 2)
        top = int(y - h / 2)
        right = int(x + w / 2)
        bottom = int(y + h / 2)
        labels.append([cls, left, top, right, bottom])
    return labels


def _xyxy_to_xywh(label: list[int], img_w: int, img_h: int) -> list[float]:
    """Convert absolute xyxy to YOLO normalized xywh."""
    cls, x1, y1, x2, y2 = label
    w = x2 - x1
    h = y2 - y1
    x_cen = round((x1 + w / 2) / img_w, 6)
    y_cen = round((y1 + h / 2) / img_h, 6)
    w_norm = round(w / img_w, 6)
    h_norm = round(h / img_h, 6)
    return [cls, x_cen, y_cen, w_norm, h_norm]


def tile_image(
    image_path: str | Path,
    output_dir: str | Path,
    patch_width: int = 640,
    patch_height: int = 640,
    overlap_factor: float = 0.5,
    label_path: str | Path | None = None,
) -> TilingMetadata:
    """Tile a single image into overlapping patches.

    If *label_path* is provided, YOLO-format labels are also sliced per-patch.
    A ``tiling_meta.json`` file is written to *output_dir* recording the original
    image dimensions so the reconstructor doesn't need hardcoded resolution.
    """
    image_path = Path(image_path)
    output_dir = Path(output_dir)
    img_dir = output_dir / "images"
    lbl_dir = output_dir / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    img_h, img_w = img.shape[:2]

    if img_h < patch_height or img_w < patch_width:
        logger.warning("Image %s (%dx%d) smaller than patch size — skipping tiling", image_path.name, img_w, img_h)
        return TilingMetadata(image_path.stem, img_h, img_w, patch_height, patch_width, overlap_factor, 0)

    step_w = int(patch_width * (1 - overlap_factor))
    step_h = int(patch_height * (1 - overlap_factor))

    # Load labels if provided
    all_labels: list[list[int]] = []
    if label_path is not None:
        label_path = Path(label_path)
        if label_path.exists():
            with open(label_path) as f:
                all_labels = _xywh_to_xyxy(f.readlines(), img_h, img_w)

    fname = image_path.stem
    idx = 0
    y_start = 0

    while y_start + patch_height <= img_h:
        x_start = 0
        while x_start + patch_width <= img_w:
            x_end = x_start + patch_width
            y_end = y_start + patch_height

            patch = img[y_start:y_end, x_start:x_end]
            cv2.imwrite(str(img_dir / f"{fname}_{idx}.png"), patch)

            # Slice labels into this patch
            if all_labels:
                cur_labels = []
                for lbl in all_labels:
                    if lbl[1] > x_start and lbl[2] > y_start and lbl[3] < x_end and lbl[4] < y_end:
                        shifted = [lbl[0], lbl[1] - x_start, lbl[2] - y_start, lbl[3] - x_start, lbl[4] - y_start]
                        cur_labels.append(_xyxy_to_xywh(shifted, patch_width, patch_height))
                if cur_labels:
                    with open(lbl_dir / f"{fname}_{idx}.txt", "w") as f:
                        for cl in cur_labels:
                            f.write(f"{cl[0]} {cl[1]} {cl[2]} {cl[3]} {cl[4]}\n")

            x_start += step_w
            idx += 1
        y_start += step_h

    # Write metadata for reconstruction
    meta = TilingMetadata(fname, img_h, img_w, patch_height, patch_width, overlap_factor, idx)
    meta_path = output_dir / "tiling_meta.json"

    # Append to existing meta file if present (for multi-image tiling)
    existing: list[dict] = []
    if meta_path.exists():
        with open(meta_path) as f:
            existing = json.load(f)
    existing.append(meta._asdict())
    with open(meta_path, "w") as f:
        json.dump(existing, f, indent=2)

    logger.info("Tiled %s → %d patches (%dx%d, overlap=%.0f%%)", image_path.name, idx, patch_width, patch_height, overlap_factor * 100)
    return meta


def tile_directory(
    image_dir: str | Path,
    output_dir: str | Path,
    patch_width: int = 640,
    patch_height: int = 640,
    overlap_factor: float = 0.5,
    label_dir: str | Path | None = None,
) -> list[TilingMetadata]:
    """Tile all images in a directory."""
    image_dir = Path(image_dir)
    results = []
    for img_path in sorted(image_dir.glob("*.png")):
        lbl_path = None
        if label_dir is not None:
            lbl_path = Path(label_dir) / f"{img_path.stem}.txt"
        meta = tile_image(img_path, output_dir, patch_width, patch_height, overlap_factor, lbl_path)
        results.append(meta)
    return results
