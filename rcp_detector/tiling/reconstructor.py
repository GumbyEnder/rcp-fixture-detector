"""Reconstruct full-resolution images and detections from tiled patches.

Refactored from Reconstructiing_less.py — reads tiling metadata instead of
hardcoding resolution to (8320, 11520).
"""

import json
import logging
from pathlib import Path

import numpy as np

from rcp_detector.detection.nms import non_max_suppression

logger = logging.getLogger(__name__)


def _load_tiling_metadata(meta_path: Path) -> dict[str, dict]:
    """Load tiling metadata, keyed by image name."""
    with open(meta_path) as f:
        entries = json.load(f)
    return {e["image_name"]: e for e in entries}


def reconstruct_detections(
    tiles_dir: str | Path,
    labels_dir: str | Path,
    output_dir: str | Path,
    meta_path: str | Path | None = None,
    patch_size: tuple[int, int] = (640, 640),
    overlap_factor: float = 0.5,
    nms_threshold: float = 0.3,
    original_resolution: tuple[int, int] | None = None,
) -> dict[str, list[dict]]:
    """Merge per-tile detection labels back into full-image coordinates.

    If *meta_path* is provided, original image dimensions are read from the
    tiling metadata (fixing the hardcoded resolution bug). Otherwise falls
    back to *original_resolution*.

    Returns a dict mapping image names to lists of detection dicts.
    """
    tiles_dir = Path(tiles_dir)
    labels_dir = Path(labels_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load metadata if available
    meta_lookup: dict[str, dict] = {}
    if meta_path is not None:
        meta_lookup = _load_tiling_metadata(Path(meta_path))

    step_h = int(patch_size[0] * (1 - overlap_factor))
    step_w = int(patch_size[1] * (1 - overlap_factor))

    # Group tile labels by source image
    tile_groups: dict[str, list[tuple[str, int]]] = {}
    for label_file in sorted(labels_dir.glob("*.txt")):
        name = label_file.stem
        if "_" not in name:
            continue
        orig_name, idx_str = name.rsplit("_", 1)
        try:
            idx = int(idx_str)
        except ValueError:
            continue
        tile_groups.setdefault(orig_name, []).append((label_file.name, idx))

    all_results: dict[str, list[dict]] = {}

    for orig_name, tiles in tile_groups.items():
        # Determine original resolution
        if orig_name in meta_lookup:
            meta = meta_lookup[orig_name]
            img_h = meta["image_height"]
            img_w = meta["image_width"]
        elif original_resolution is not None:
            img_h, img_w = original_resolution
        else:
            raise ValueError(
                f"No tiling metadata for '{orig_name}' and no original_resolution fallback provided."
            )

        num_patches_w = (img_w - patch_size[1]) // step_w + 1

        all_detections: list[list[float]] = []

        for label_file, patch_idx in tiles:
            x_idx = patch_idx % num_patches_w
            y_idx = patch_idx // num_patches_w

            label_path = labels_dir / label_file
            with open(label_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    # Support both 5-field (no conf) and 6-field (with conf) formats
                    class_id = int(float(parts[0]))
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    confidence = float(parts[5]) if len(parts) >= 6 else 1.0

                    # Convert relative patch coords to absolute image coords
                    x1 = (x_center - width / 2) * patch_size[1] + x_idx * step_w
                    y1 = (y_center - height / 2) * patch_size[0] + y_idx * step_h
                    x2 = (x_center + width / 2) * patch_size[1] + x_idx * step_w
                    y2 = (y_center + height / 2) * patch_size[0] + y_idx * step_h

                    all_detections.append([x1, y1, x2, y2, confidence, class_id])

        if not all_detections:
            all_results[orig_name] = []
            continue

        det_array = np.array(all_detections)
        nms_dets = non_max_suppression(det_array, nms_threshold)

        # Build structured results
        detections = []
        for det in nms_dets:
            x1, y1, x2, y2, conf, cls_id = det
            detections.append({
                "class_id": int(cls_id),
                "confidence": float(conf),
                "bbox_abs": [float(x1), float(y1), float(x2), float(y2)],
                "bbox_norm": [
                    float((x1 + x2) / 2 / img_w),
                    float((y1 + y2) / 2 / img_h),
                    float((x2 - x1) / img_w),
                    float((y2 - y1) / img_h),
                ],
            })

        # Also write YOLO-format label file
        with open(output_dir / f"{orig_name}.txt", "w") as f:
            for d in detections:
                cx, cy, w, h = d["bbox_norm"]
                f.write(f"{d['class_id']} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

        all_results[orig_name] = detections
        logger.info("Reconstructed %s: %d detections after NMS", orig_name, len(detections))

    return all_results
