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


def _line_thickness(min_side: int, *, div: int = 220, lo: int = 2, hi: int = 6) -> int:
    """Scale stroke width from image shorter side (full OPG vs quadrant crop)."""
    return int(np.clip(max(lo, round(min_side / div)), lo, hi))


def _label_font_scale(min_side: int, *, div: float = 650.0, lo: float = 0.55, hi: float = 1.35) -> float:
    return float(np.clip(min_side / div, lo, hi))


def _confidence_suffix(confidence: object) -> str:
    if not isinstance(confidence, (int, float)):
        return ""
    c = float(confidence)
    if 0.0 <= c <= 1.0:
        return f"  {c * 100:.0f}%"
    return f"  {c:.2f}"


def _draw_label_panel(
    canvas: np.ndarray,
    *,
    anchor_x: int,
    anchor_y_top: int,
    text: str,
    accent_bgr: tuple[int, int, int],
    min_side: int,
) -> None:
    """Dark filled panel + light text for readable captions (diagnosis + confidence)."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = _label_font_scale(min_side)
    th = max(2, int(round(fs * 2)))
    ((tw, th_h), bl) = cv2.getTextSize(text, font, fs, th)
    pad_h, pad_v = 12, 10
    box_w = tw + 2 * pad_h
    box_h = th_h + bl + 2 * pad_v
    h_img, w_img = canvas.shape[:2]
    lx1 = int(np.clip(anchor_x, 0, w_img - 1))
    ly1 = int(np.clip(anchor_y_top, 0, h_img - 1))
    lx2 = int(np.clip(lx1 + box_w, 0, w_img - 1))
    ly2 = int(np.clip(ly1 + box_h, 0, h_img - 1))
    if lx2 <= lx1 + 4 or ly2 <= ly1 + 4:
        return
    cv2.rectangle(canvas, (lx1, ly1), (lx2, ly2), (22, 22, 22), -1)
    cv2.rectangle(canvas, (lx1, ly1), (lx2, ly2), accent_bgr, 2)
    tx = lx1 + pad_h
    ty = ly2 - pad_v - 2
    cv2.putText(canvas, text, (tx, ty), font, fs, (252, 252, 252), th, cv2.LINE_AA)


def _place_caption_above_box(
    canvas: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    label: str,
    confidence: object,
    accent_bgr: tuple[int, int, int],
    min_side: int,
) -> None:
    """Place diagnosis + confidence in a high-contrast panel (above box, or below if no room)."""
    text = f"{label}{_confidence_suffix(confidence)}"
    fs = _label_font_scale(min_side)
    th = max(2, int(round(fs * 2)))
    font = cv2.FONT_HERSHEY_SIMPLEX
    ((_, th_h), bl) = cv2.getTextSize(text, font, fs, th)
    pad_h, pad_v = 12, 10
    box_h = th_h + bl + 2 * pad_v
    h_img = canvas.shape[0]
    ly1 = y1 - box_h - 6
    if ly1 < 2:
        ly1 = min(h_img - box_h - 2, y2 + 6)
    _draw_label_panel(
        canvas,
        anchor_x=x1,
        anchor_y_top=ly1,
        text=text,
        accent_bgr=accent_bgr,
        min_side=min_side,
    )


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
    h, w = orig_img.shape[:2]
    min_side = min(h, w)
    lt = _line_thickness(min_side, div=200, lo=2, hi=5)

    if result.boxes is None or len(result.boxes) == 0:
        return orig_img

    boxes = result.boxes.xyxy.cpu().numpy().astype(np.int32)
    classes = result.boxes.cls.cpu().numpy().astype(np.int32)
    confs = result.boxes.conf.cpu().numpy().tolist() if result.boxes.conf is not None else [None] * len(boxes)
    num_classes = int(classes.max()) + 1 if classes.size > 0 else 1
    class_to_color = _class_palette(num_classes)
    masks = result.masks.xy if result.masks is not None else [None] * len(boxes)

    overlay = orig_img.copy()
    for idx, (mask, cls_id, box) in enumerate(zip(masks, classes, boxes)):
        color = class_to_color.get(int(cls_id), (255, 255, 255))
        if mask is not None:
            points = np.array(mask, dtype=np.int32)
            cv2.fillPoly(overlay, [points], color)

    blended = cv2.addWeighted(overlay, alpha, orig_img, 1 - alpha, 0)
    out = blended.copy()

    for idx, (mask, cls_id, box) in enumerate(zip(masks, classes, boxes)):
        color = class_to_color.get(int(cls_id), (255, 255, 255))
        x1, y1, x2, y2 = box
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, lt)
        label = custom_labels[idx] if custom_labels and idx < len(custom_labels) else f"{label_prefix}{int(cls_id)}"
        conf = confs[idx] if idx < len(confs) else None
        _place_caption_above_box(out, x1, y1, x2, y2, label, conf, color, min_side)
        if mask is not None:
            points = np.array(mask, dtype=np.int32)
            cv2.polylines(out, [points], True, color, max(2, lt - 1), lineType=cv2.LINE_AA)
            cv2.polylines(out, [points], True, (255, 255, 255), 1, lineType=cv2.LINE_AA)

    return out


def draw_global_detections_overlay(
    base_image_bgr: np.ndarray,
    detections: List[dict],
    alpha: float = 0.35,
) -> np.ndarray:
    """Draw pre-remapped global detections on the full image.

    Fills are blended for readability; **boxes, mask outlines, and labels** are drawn
    again on top at full opacity so diagnosis names, borders, and confidence stay legible.
    """
    h, w = base_image_bgr.shape[:2]
    min_side = min(h, w)
    lt = _line_thickness(min_side, div=180, lo=3, hi=8)

    def _accent(det: dict, cls_id: int) -> tuple[int, int, int]:
        lab = str(det.get("label", "")).lower()
        if "caries" in lab:
            return (50, 50, 255)
        if "impact" in lab:
            return (255, 140, 40)
        palette = _class_palette(16)
        return palette.get(cls_id % 16, (255, 255, 255))

    overlay = base_image_bgr.copy()
    for det in detections:
        cls_id = int(det.get("class_id", 0))
        color = _accent(det, cls_id)
        polygon = det.get("mask_xy")
        if polygon:
            points = np.array(polygon, dtype=np.int32)
            cv2.fillPoly(overlay, [points], color)
    blended = cv2.addWeighted(overlay, alpha, base_image_bgr, 1 - alpha, 0)
    out = blended.copy()

    for det in detections:
        cls_id = int(det.get("class_id", 0))
        color = _accent(det, cls_id)
        box = det.get("box_xyxy")
        polygon = det.get("mask_xy")
        x1 = y1 = x2 = y2 = None
        if box and len(box) == 4:
            x1, y1, x2, y2 = [int(v) for v in box]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)
        elif polygon:
            arr = np.array(polygon, dtype=np.float32)
            x1 = int(np.clip(arr[:, 0].min(), 0, w - 1))
            y1 = int(np.clip(arr[:, 1].min(), 0, h - 1))
            x2 = int(np.clip(arr[:, 0].max(), 0, w - 1))
            y2 = int(np.clip(arr[:, 1].max(), 0, h - 1))
        if x1 is not None and x2 is not None and x2 > x1 and y2 > y1:
            cv2.rectangle(out, (x1, y1), (x2, y2), color, lt)
            _place_caption_above_box(
                out,
                x1,
                y1,
                x2,
                y2,
                str(det.get("label", str(cls_id))),
                det.get("confidence"),
                color,
                min_side,
            )
        if polygon:
            points = np.array(polygon, dtype=np.int32)
            cv2.polylines(out, [points], True, color, max(2, lt - 1), lineType=cv2.LINE_AA)
            cv2.polylines(out, [points], True, (255, 255, 255), 1, lineType=cv2.LINE_AA)

    return out


def draw_quadrant_crop_pathology_overlay(
    crop_bgr: np.ndarray,
    *,
    top_left: tuple[int, int],
    quadrant_id: int,
    periapical: List[dict],
    teeth_classification: List[dict],
    alpha: float = 0.42,
) -> np.ndarray:
    """Draw periapical + teeth-classification (e.g. caries, impacted) on one quadrant crop in local coordinates.

    Expects ``box_xyxy`` / ``mask_xy`` in **full-image** coordinates; subtracts ``top_left`` to map onto the crop.
    Mask fills are blended; **boxes, contours, and label panels** are drawn on top at full opacity for legibility.
    """
    px1, py1 = int(top_left[0]), int(top_left[1])
    h, w = crop_bgr.shape[:2]
    min_side = min(h, w)
    lt = _line_thickness(min_side, div=140, lo=2, hi=5)
    base = crop_bgr.copy()
    overlay = base.copy()

    def _color_for(det: dict) -> tuple[int, int, int]:
        lab = str(det.get("label", "")).lower()
        if "caries" in lab:
            return (50, 50, 255)
        if "impact" in lab:
            return (255, 140, 40)
        return (60, 220, 255)

    for det in list(periapical) + list(teeth_classification):
        if int(det.get("quadrant_id", -1)) != int(quadrant_id):
            continue
        color = _color_for(det)
        poly = det.get("mask_xy")
        if poly and len(poly) >= 3:
            pts = np.array([[float(x) - px1, float(y) - py1] for x, y in poly], dtype=np.float32)
            pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
            pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
            pts_i = np.round(pts).astype(np.int32)
            cv2.fillPoly(overlay, [pts_i], color)

    out = cv2.addWeighted(overlay, alpha, base, 1 - alpha, 0)

    for det in list(periapical) + list(teeth_classification):
        if int(det.get("quadrant_id", -1)) != int(quadrant_id):
            continue
        color = _color_for(det)
        box = det.get("box_xyxy")
        poly = det.get("mask_xy")
        x1 = y1 = x2 = y2 = None
        if box and len(box) == 4:
            gx1, gy1, gx2, gy2 = (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
            x1 = int(max(0, min(w - 1, gx1 - px1)))
            y1 = int(max(0, min(h - 1, gy1 - py1)))
            x2 = int(max(0, min(w - 1, gx2 - px1)))
            y2 = int(max(0, min(h - 1, gy2 - py1)))
        elif poly and len(poly) >= 3:
            pts = np.array([[float(x) - px1, float(y) - py1] for x, y in poly], dtype=np.float32)
            pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
            pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
            x1 = int(pts[:, 0].min())
            y1 = int(pts[:, 1].min())
            x2 = int(pts[:, 0].max())
            y2 = int(pts[:, 1].max())
        if x1 is not None and x2 is not None and x2 > x1 and y2 > y1:
            cv2.rectangle(out, (x1, y1), (x2, y2), color, lt)
            _place_caption_above_box(
                out,
                x1,
                y1,
                x2,
                y2,
                str(det.get("label", "")),
                det.get("confidence"),
                color,
                min_side,
            )
        if poly and len(poly) >= 3:
            pts = np.array([[float(x) - px1, float(y) - py1] for x, y in poly], dtype=np.float32)
            pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
            pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
            pts_i = np.round(pts).astype(np.int32)
            cv2.polylines(out, [pts_i], True, color, max(2, lt - 1), lineType=cv2.LINE_AA)
            cv2.polylines(out, [pts_i], True, (255, 255, 255), 1, lineType=cv2.LINE_AA)

    return out


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
