"""YOLOv8 inference via the ultralytics API — replaces vendored detect.py."""

import logging
from pathlib import Path

from ultralytics import YOLO

logger = logging.getLogger(__name__)


class FixtureDetector:
    """Thin wrapper around ultralytics YOLO for fixture detection."""

    def __init__(
        self,
        model_path: str | Path = "yolov8l.pt",
        confidence: float = 0.25,
        iou: float = 0.45,
        device: str = "",
        imgsz: int = 640,
    ):
        self.model = YOLO(str(model_path))
        self.confidence = confidence
        self.iou = iou
        self.device = device or None
        self.imgsz = imgsz

    def predict_tiles(
        self,
        tiles_dir: str | Path,
        output_dir: str | Path,
        batch_size: int = 16,
        save_labels: bool = True,
        save_images: bool = False,
    ) -> list[dict]:
        """Run inference on a directory of tile images.

        Writes YOLO-format label files (with confidence) to *output_dir*/labels/.
        Returns a list of per-tile result dicts.
        """
        tiles_dir = Path(tiles_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        labels_dir = output_dir / "labels"
        labels_dir.mkdir(exist_ok=True)

        image_files = sorted(
            p for p in tiles_dir.iterdir()
            if p.suffix.lower() in (".png", ".jpg", ".jpeg")
        )

        if not image_files:
            logger.warning("No images found in %s", tiles_dir)
            return []

        logger.info("Running inference on %d tiles (batch=%d)", len(image_files), batch_size)

        all_results = []

        # Process in batches to avoid OOM — use stream=True for memory efficiency
        for batch_start in range(0, len(image_files), batch_size):
            batch_files = image_files[batch_start:batch_start + batch_size]
            results_iter = self.model.predict(
                source=[str(p) for p in batch_files],
                conf=self.confidence,
                iou=self.iou,
                device=self.device,
                imgsz=self.imgsz,
                batch=batch_size,
                save=save_images,
                save_txt=False,
                stream=True,
                verbose=False,
            )

            for img_path, result in zip(batch_files, results_iter):
                detections = []
                if result.boxes is not None and len(result.boxes):
                    boxes = result.boxes
                    label_path = labels_dir / f"{img_path.stem}.txt"
                    with open(label_path, "w") as f:
                        for i in range(len(boxes)):
                            cls_id = int(boxes.cls[i].item())
                            conf = float(boxes.conf[i].item())
                            xywhn = boxes.xywhn[i].tolist()
                            f.write(f"{cls_id} {xywhn[0]:.6f} {xywhn[1]:.6f} {xywhn[2]:.6f} {xywhn[3]:.6f} {conf:.6f}\n")
                            detections.append({
                                "class_id": cls_id,
                                "confidence": conf,
                                "xywhn": xywhn,
                            })

                all_results.append({
                    "tile": img_path.name,
                    "num_detections": len(detections),
                    "detections": detections,
                })

            if batch_start % (batch_size * 10) == 0:
                done = min(batch_start + batch_size, len(image_files))
                logger.info("Progress: %d/%d tiles", done, len(image_files))

        total = sum(r["num_detections"] for r in all_results)
        logger.info("Inference complete: %d total detections across %d tiles", total, len(all_results))
        return all_results
