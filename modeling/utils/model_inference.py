from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from .common import ImagePath, apply_clahe, to_path

DEFAULT_MODEL_PATHS: Dict[str, str] = {
    "quadrants": "models/quadrant segmentation/best.pt",
    "teeth": "models/teeth segmentation/best.pt",
    "periapical": "models/periapical detector (cropped)/best.pt",
    "teeth_classification": "models/teeth classification/best.pt",
}


class ModelRegistry:
    """Lazy YOLO model loader to avoid reloading weights repeatedly."""

    def __init__(self, model_paths: Optional[Dict[str, str]] = None) -> None:
        self.model_paths = model_paths or DEFAULT_MODEL_PATHS.copy()
        self._cache: Dict[str, YOLO] = {}

    def get(self, key: str) -> YOLO:
        """Return a cached YOLO model for the given key."""
        if key not in self.model_paths:
            raise KeyError(f"Model key '{key}' not defined. Available: {list(self.model_paths.keys())}")
        if key not in self._cache:
            model_path = to_path(self.model_paths[key])
            if not model_path.exists():
                raise FileNotFoundError(f"Model not found for '{key}': {model_path}")
            self._cache[key] = YOLO(str(model_path))
        return self._cache[key]


def segment_quadrants(
    image_path: ImagePath,
    model_registry: ModelRegistry,
    conf: float = 0.5,
):
    """Segment image quadrants and keep the highest confidence prediction per class [0..3]."""
    clahe_img = apply_clahe(image_path)
    clahe_bgr = cv2.cvtColor(clahe_img, cv2.COLOR_GRAY2BGR)
    model = model_registry.get("quadrants")

    results = model.predict(source=clahe_bgr, conf=conf, verbose=False)
    result = results[0]

    keep_indices = []
    for cls in [0, 1, 2, 3]:
        indices = torch.where(result.boxes.cls == cls)[0]
        if len(indices) > 0:
            class_confs = result.boxes.conf[indices]
            best_idx = indices[torch.argmax(class_confs)].item()
            keep_indices.append(best_idx)

    if not keep_indices:
        return result
    return result[keep_indices]


def get_quadrant_crops(
    image_path: ImagePath,
    model_registry: ModelRegistry,
    padding: int = 40,
    quadrant_conf: float = 0.01,
):
    """Return original image, quadrant crop metadata, and filename."""
    result = segment_quadrants(image_path=image_path, model_registry=model_registry, conf=quadrant_conf)
    orig_img = result.orig_img.copy()
    h_img, w_img = orig_img.shape[:2]
    filename = Path(image_path).name

    if result.boxes is None or len(result.boxes) == 0:
        return orig_img, [], filename

    boxes = result.boxes.xyxy.cpu().numpy().astype(np.int32)
    classes = result.boxes.cls.cpu().numpy().astype(np.int32)
    masks_xy = result.masks.xy if result.masks is not None else [None] * len(boxes)

    crops_data = []
    for box, cls_id, m_xy in zip(boxes, classes, masks_xy):
        x1, y1, x2, y2 = box
        px1, py1 = max(0, x1 - padding), max(0, y1 - padding)
        px2, py2 = min(w_img, x2 + padding), min(h_img, y2 + padding)
        crop = orig_img[py1:py2, px1:px2].copy()

        crops_data.append(
            {
                "crop": crop,
                "class_id": int(cls_id),
                "mask_xy": m_xy,
                "original_box": [int(x1), int(y1), int(x2), int(y2)],
                "top_left": (int(px1), int(py1)),
            }
        )

    return orig_img, crops_data, filename


def segment_teeth(
    image_path: ImagePath,
    model_registry: ModelRegistry,
    conf: float = 0.5,
):
    """Run teeth segmentation model and return first Ultralytics result."""
    model = model_registry.get("teeth")
    results = model.predict(source=str(to_path(image_path)), conf=conf, verbose=False)
    return results[0]


def segment_teeth_on_array(
    image_bgr: np.ndarray,
    model_registry: ModelRegistry,
    conf: float = 0.5,
):
    """Run teeth segmentation on an in-memory BGR image."""
    model = model_registry.get("teeth")
    results = model.predict(source=image_bgr, conf=conf, verbose=False)
    return results[0]


def detect_periapical(
    image_path: ImagePath,
    model_registry: ModelRegistry,
    conf: float = 0.3,
):
    """Run periapical detector and return first Ultralytics result."""
    model = model_registry.get("periapical")
    results = model.predict(source=str(to_path(image_path)), conf=conf, verbose=False)
    return results[0]


def detect_periapical_on_array(
    image_bgr: np.ndarray,
    model_registry: ModelRegistry,
    conf: float = 0.3,
):
    """Run periapical detection on an in-memory BGR image."""
    model = model_registry.get("periapical")
    results = model.predict(source=image_bgr, conf=conf, verbose=False)
    return results[0]


def classify_teeth_on_array(
    image_bgr: np.ndarray,
    model_registry: ModelRegistry,
    conf: float = 0.3,
):
    """Run teeth classification / pathology segmentation on a quadrant crop (BGR)."""
    model = model_registry.get("teeth_classification")
    results = model.predict(source=image_bgr, conf=conf, verbose=False)
    return results[0]
