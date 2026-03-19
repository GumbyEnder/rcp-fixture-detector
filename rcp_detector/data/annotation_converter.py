"""Convert JSON annotations to YOLO format.

Merged from code.py + Covert_Json_to_txt.py — config-driven, no hardcoded paths.
"""

import json
import logging
from pathlib import Path

import cv2

logger = logging.getLogger(__name__)


def load_class_names(classes_file: str | Path) -> dict[str, int]:
    """Load class names from a text file, returning name→index mapping."""
    with open(classes_file) as f:
        names = [line.strip() for line in f if line.strip()]
    return {name: i for i, name in enumerate(names)}


def convert_json_to_yolo(
    input_dir: str | Path,
    output_dir: str | Path,
    classes_file: str | Path,
    copy_images: bool = True,
) -> int:
    """Convert JSON annotation files to YOLO format .txt files.

    Expects JSON files with structure: [{annotations: [{class, x, y, width, height}, ...]}]
    alongside corresponding .png images.

    Returns the number of files converted.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    class_dict = load_class_names(classes_file)
    json_files = sorted(input_dir.glob("*.json"))
    count = 0

    for json_file in json_files:
        with open(json_file) as f:
            json_content = json.load(f)

        image_name = json_file.stem + ".png"
        image_path = input_dir / image_name

        img = cv2.imread(str(image_path))
        if img is None:
            logger.warning("Image not found for %s — skipping", json_file.name)
            continue

        img_h, img_w = img.shape[:2]

        label_path = output_dir / (json_file.stem + ".txt")
        with open(label_path, "w") as out_f:
            for ann in json_content[0]["annotations"]:
                class_name = ann["class"].rstrip()
                if class_name not in class_dict:
                    logger.warning("Unknown class '%s' in %s — skipping", class_name, json_file.name)
                    continue
                class_idx = class_dict[class_name]

                x, y, w, h = ann["x"], ann["y"], ann["width"], ann["height"]
                x_center = (x + w / 2) / img_w
                y_center = (y + h / 2) / img_h
                norm_w = w / img_w
                norm_h = h / img_h

                out_f.write(f"{class_idx} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}\n")

        if copy_images:
            cv2.imwrite(str(output_dir / image_name), img)

        count += 1

    logger.info("Converted %d JSON annotations to YOLO format", count)
    return count
