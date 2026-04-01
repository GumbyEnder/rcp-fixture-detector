
from __future__ import annotations

from pathlib import Path

import cv2


def draw_detections(image_path: str | Path, detections: list[dict], viz_path: str | Path, class_names: dict[int, str], line_width: int = 2) -> None:
    """Draw YOLO-style detections onto an image and save the preview."""
    image_path = Path(image_path)
    viz_path = Path(viz_path)
    viz_path.parent.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    for det in detections:
        bbox = det.get("bbox_abs") or det.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = map(int, bbox)
        class_id = int(det.get("class_id", -1))
        label = class_names.get(class_id, f"class_{class_id}")
        confidence = det.get("confidence")
        if confidence is not None:
            label = f"{label} {confidence:.2f}"
        color = (0, 255, 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, line_width)
        cv2.putText(img, label, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, max(1, line_width - 1), cv2.LINE_AA)

    cv2.imwrite(str(viz_path), img)
