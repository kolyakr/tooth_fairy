from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import cv2
import matplotlib
import matplotlib.gridspec as gridspec
import numpy as np

# Force headless backend for background-thread inference on macOS.
matplotlib.use("Agg")
import matplotlib.pyplot as plt

QUADRANT_COLORS = [(0, 255, 0), (0, 165, 255), (255, 150, 0), (255, 0, 255)]


def _class_palette(num_classes: int) -> Dict[int, tuple]:
    palette = {}
    hues = np.linspace(0, 180, max(num_classes, 1), endpoint=False)
    for cls in range(num_classes):
        hsv_color = np.uint8([[[hues[cls], 200, 255]]])
        bgr = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0]
        palette[cls] = (int(bgr[0]), int(bgr[1]), int(bgr[2]))
    return palette


def draw_quadrants_overlay(processed_image: np.ndarray, crops_data: List[dict], alpha: float = 0.3) -> np.ndarray:
    """Draw quadrant masks, boxes, and labels over an image."""
    overlay = processed_image.copy()
    for item in crops_data:
        cls_id = item["class_id"]
        color = QUADRANT_COLORS[cls_id] if cls_id < len(QUADRANT_COLORS) else (255, 255, 255)
        if item["mask_xy"] is not None:
            pts = np.array(item["mask_xy"], dtype=np.int32)
            cv2.fillPoly(overlay, [pts], color)
        x1, y1, x2, y2 = item["original_box"]
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 3)
        cv2.putText(overlay, f"Q{cls_id}", (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
    return cv2.addWeighted(overlay, alpha, processed_image, 1 - alpha, 0)


def draw_instances_overlay(result, alpha: float = 0.35, label_prefix: str = "", custom_labels: List[str] | None = None) -> np.ndarray:
    """Draw mask/box overlays for a generic Ultralytics result."""
    orig_img = result.orig_img.copy()
    overlay = orig_img.copy()

    if result.boxes is None or len(result.boxes) == 0:
        return orig_img

    boxes = result.boxes.xyxy.cpu().numpy().astype(np.int32)
    classes = result.boxes.cls.cpu().numpy().astype(np.int32)
    num_classes = int(classes.max()) + 1 if classes.size > 0 else 1
    class_to_color = _class_palette(num_classes)
    masks = result.masks.xy if result.masks is not None else [None] * len(boxes)

    for idx, (mask, cls_id, box) in enumerate(zip(masks, classes, boxes)):
        color = class_to_color.get(int(cls_id), (255, 255, 255))
        if mask is not None:
            points = np.array(mask, dtype=np.int32)
            cv2.fillPoly(overlay, [points], color)
            cv2.polylines(overlay, [points], True, (255, 255, 255), 1)

        x1, y1, x2, y2 = box
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        label = custom_labels[idx] if custom_labels and idx < len(custom_labels) else f"{label_prefix}{int(cls_id)}"
        cv2.putText(
            overlay,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
            cv2.LINE_AA,
        )

    return cv2.addWeighted(overlay, alpha, orig_img, 1 - alpha, 0)


def draw_global_detections_overlay(
    base_image_bgr: np.ndarray,
    detections: List[dict],
    alpha: float = 0.35,
) -> np.ndarray:
    """Draw pre-remapped global detections on the full image."""
    overlay = base_image_bgr.copy()
    class_to_color = _class_palette(16)

    for det in detections:
        cls_id = int(det.get("class_id", 0))
        color = class_to_color.get(cls_id % 16, (255, 255, 255))
        box = det.get("box_xyxy")
        if box:
            x1, y1, x2, y2 = [int(v) for v in box]
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
            label = det.get("label", str(cls_id))
            cv2.putText(
                overlay,
                label,
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
                cv2.LINE_AA,
            )

        polygon = det.get("mask_xy")
        if polygon:
            points = np.array(polygon, dtype=np.int32)
            cv2.fillPoly(overlay, [points], color)
            cv2.polylines(overlay, [points], True, (255, 255, 255), 1)

    return cv2.addWeighted(overlay, alpha, base_image_bgr, 1 - alpha, 0)


def save_bgr(path: Path, image_bgr: np.ndarray) -> None:
    """Save a BGR image to disk, raising on failure."""
    ok = cv2.imwrite(str(path), image_bgr)
    if not ok:
        raise RuntimeError(f"Could not write image to: {path}")


def save_quadrant_grid(path: Path, processed_image: np.ndarray, crops_data: List[dict], filename: str, alpha: float = 0.3) -> None:
    """Save a figure with global quadrant overlay and four crop panels."""
    fig = plt.figure(figsize=(16, 18))
    gs = gridspec.GridSpec(3, 2, height_ratios=[1.2, 1, 1])

    combined = draw_quadrants_overlay(processed_image=processed_image, crops_data=crops_data, alpha=alpha)
    ax0 = fig.add_subplot(gs[0, :])
    ax0.imshow(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))
    ax0.set_title(filename, fontsize=20, fontweight="bold")
    ax0.axis("off")

    grid_pos = {0: (1, 0), 1: (1, 1), 2: (2, 0), 3: (2, 1)}
    for item in crops_data:
        cls_id = item["class_id"]
        if cls_id not in grid_pos:
            continue
        crop = item["crop"].copy()
        row, col = grid_pos[cls_id]
        ax = fig.add_subplot(gs[row, col])
        ax.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        ax.set_title(f"Quadrant {cls_id}", fontsize=14, color="darkblue")
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
