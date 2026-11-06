from pathlib import Path
from time import perf_counter
from typing import Any

import cv2
import numpy as np


class ModelUnavailableError(RuntimeError):
    """The configured pretrained segmentation weights are not installed."""


_model_cache: dict[Path, Any] = {}


def get_model(model_path: Path) -> Any:
    resolved_path = model_path.resolve()
    if not resolved_path.is_file():
        raise ModelUnavailableError(
            f"YOLO segmentation model not found at {resolved_path}. "
            "Download yolo11n-seg.pt and place it there, or set SHADOWVISION_MODEL."
        )
    if resolved_path not in _model_cache:
        from ultralytics import YOLO

        _model_cache[resolved_path] = YOLO(str(resolved_path))
    return _model_cache[resolved_path]


def run_segmentation(
    image: np.ndarray,
    model_path: Path,
    confidence: float = 0.25,
) -> dict[str, Any]:
    if not 0.05 <= confidence <= 0.95:
        raise ValueError("Confidence threshold must be between 0.05 and 0.95.")

    model = get_model(model_path)
    started = perf_counter()
    result = model.predict(source=image, conf=confidence, verbose=False)[0]
    elapsed = round(perf_counter() - started, 4)

    annotated = result.plot()
    detections: list[dict[str, Any]] = []
    if result.boxes is not None:
        boxes = result.boxes
        coordinates = boxes.xyxy.cpu().numpy()
        scores = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)
        for box, score, class_id in zip(coordinates, scores, classes):
            detections.append(
                {
                    "class_id": int(class_id),
                    "label": str(result.names[int(class_id)]),
                    "confidence": round(float(score), 4),
                    "box": [round(float(value), 1) for value in box],
                }
            )

    masks: list[np.ndarray] = []
    if result.masks is not None:
        for raw_mask in result.masks.data.cpu().numpy():
            resized = cv2.resize(
                raw_mask.astype(np.float32),
                (image.shape[1], image.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
            masks.append((resized > 0.5).astype(np.uint8) * 255)

    return {
        "annotated": annotated,
        "detections": detections,
        "masks": masks,
        "processing_time": elapsed,
    }
