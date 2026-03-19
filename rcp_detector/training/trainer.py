"""YOLOv8 training wrapper via ultralytics API."""

import logging
from pathlib import Path

from ultralytics import YOLO

logger = logging.getLogger(__name__)


def train(
    data_yaml: str | Path,
    model_path: str | Path = "yolov8l.pt",
    epochs: int = 100,
    batch_size: int = 16,
    imgsz: int = 640,
    patience: int = 20,
    device: str = "",
    project: str = "runs/train",
    name: str = "rcp",
    **kwargs,
) -> Path:
    """Train a YOLOv8 model with transfer learning.

    Returns the path to the best weights.
    """
    model = YOLO(str(model_path))

    logger.info(
        "Starting training: data=%s, epochs=%d, batch=%d, imgsz=%d",
        data_yaml, epochs, batch_size, imgsz,
    )

    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        patience=patience,
        device=device or None,
        project=project,
        name=name,
        exist_ok=True,
        **kwargs,
    )

    best_weights = Path(project) / name / "weights" / "best.pt"
    logger.info("Training complete. Best weights: %s", best_weights)
    return best_weights
