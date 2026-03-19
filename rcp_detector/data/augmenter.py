"""Data augmentation for YOLO datasets with class-aware balancing.

Refactored from augmentation4B_main5.py — no hardcoded paths.
"""

import logging
import os
import random
from pathlib import Path

import albumentations as A
import cv2
from tqdm import tqdm

logger = logging.getLogger(__name__)


def _yolo_to_pascal(size: tuple[int, int], box: list[float]) -> list[float]:
    w_img, h_img = size
    x, y, w, h = box
    x *= w_img
    y *= h_img
    w *= w_img
    h *= h_img
    return [x - w / 2, y - h / 2, x + w / 2, y + h / 2]


def _pascal_to_yolo(size: tuple[int, int], box: list[float]) -> list[float]:
    w_img, h_img = size
    x = (box[0] + box[2]) / 2.0 / w_img
    y = (box[1] + box[3]) / 2.0 / h_img
    w = (box[2] - box[0]) / w_img
    h = (box[3] - box[1]) / h_img
    return [x, y, w, h]


def augment_dataset(
    data_dir: str | Path,
    output_dir: str | Path,
    classes_file: str | Path | None = None,
    target_size: int = 640,
    seed: int = 42,
) -> int:
    """Augment images to balance class distribution.

    Auto-determines augmentation targets:
    - Classes with <100 samples → 4x
    - Classes with <1000 samples → 2x
    - Classes with >=1000 samples → no augmentation

    Returns total augmented images created.
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(seed)

    transform = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.RandomScale(scale_limit=0.1, p=0.5, interpolation=cv2.INTER_LINEAR),
            A.Rotate(limit=5, p=0.3, border_mode=cv2.BORDER_CONSTANT),
            A.GaussNoise(p=0.2),
            A.Resize(target_size, target_size),
        ],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["class_labels"]),
    )

    # Count existing class distribution
    class_counts: dict[int, int] = {}
    label_files = sorted(data_dir.glob("*.txt"))
    for lf in label_files:
        with open(lf) as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    cls_id = int(parts[0])
                    class_counts[cls_id] = class_counts.get(cls_id, 0) + 1

    # Determine augmentation thresholds
    thresholds: dict[int, int] = {}
    for cls_id, count in class_counts.items():
        if count < 100:
            thresholds[cls_id] = count * 4
        elif count < 1000:
            thresholds[cls_id] = count * 2

    if not thresholds:
        logger.info("All classes have >=1000 samples — no augmentation needed")
        return 0

    image_files = sorted(data_dir.glob("*.png"))
    random.shuffle(image_files)

    aug_counts = dict(class_counts)
    created = 0

    pbar = tqdm(total=sum(thresholds.values()), desc="Augmenting")

    for img_path in image_files:
        if not thresholds:
            break

        label_path = img_path.with_suffix(".txt")
        if not label_path.exists():
            continue

        with open(label_path) as f:
            lines = f.read().splitlines()

        # Check if this image has any class needing augmentation
        bboxes = []
        class_labels = []
        image = None

        for line in lines:
            parts = line.split()
            cls_id = int(parts[0])
            if cls_id in thresholds and aug_counts.get(cls_id, 0) < thresholds[cls_id]:
                if image is None:
                    image = cv2.imread(str(img_path))
                    if image is None:
                        break
                h, w = image.shape[:2]
                box = list(map(float, parts[1:5]))
                bboxes.append(_yolo_to_pascal((w, h), box))
                class_labels.append(cls_id)

        if not bboxes or image is None:
            continue

        augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)

        base_name = img_path.stem + f"_aug{created}"
        new_img = output_dir / f"{base_name}.png"
        new_lbl = output_dir / f"{base_name}.txt"

        cv2.imwrite(str(new_img), augmented["image"])
        with open(new_lbl, "w") as f:
            for box, label in zip(augmented["bboxes"], augmented["class_labels"]):
                yolo_box = _pascal_to_yolo((w, h), list(box))
                f.write(f"{label} {' '.join(f'{v:.6f}' for v in yolo_box)}\n")
                aug_counts[label] = aug_counts.get(label, 0) + 1
                if label in thresholds and aug_counts[label] >= thresholds[label]:
                    del thresholds[label]

        created += 1
        pbar.update(1)

    pbar.close()
    logger.info("Created %d augmented images", created)
    return created
