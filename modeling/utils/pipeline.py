from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set

from .common import ImagePath, ensure_dir, read_image_bgr, to_path
from .model_inference import (
    ModelRegistry,
    classify_teeth_on_array,
    detect_periapical_on_array,
    get_quadrant_crops,
    segment_teeth_on_array,
)
from .visualization import (
    draw_global_detections_overlay,
    draw_quadrants_overlay,
    save_bgr,
    save_quadrant_grid,
)

VALID_TASKS: Set[str] = {"quadrants", "teeth", "periapical", "teeth_classification"}

# Canonical labels exposed by the app for the teeth classification model.
# We currently support two classes in production: Caries and Impacted.
_DEFAULT_TEETH_CLASS_LABELS: Dict[int, str] = {
    0: "Impacted",
    1: "Caries",
}


def _normalize_teeth_class_label(raw: str) -> str:
    """Map model class names to stable frontend/backend labels."""
    normalized = raw.strip().lower().replace("_", " ").replace("-", " ")
    if "impacted" in normalized:
        return "Impacted"
    if "caries" in normalized:
        return "Caries"
    return raw.strip()


def _resolve_teeth_class_label(class_id: int, model_names: object) -> str:
    """Resolve class label from YOLO names with safe fallback."""
    raw_label: str | None = None
    if isinstance(model_names, dict):
        candidate = model_names.get(class_id)
        if isinstance(candidate, str):
            raw_label = candidate
    elif isinstance(model_names, (list, tuple)) and 0 <= class_id < len(model_names):
        candidate = model_names[class_id]
        if isinstance(candidate, str):
            raw_label = candidate

    if raw_label:
        normalized = _normalize_teeth_class_label(raw_label)
        if normalized in {"Caries", "Impacted"}:
            return normalized

    return _DEFAULT_TEETH_CLASS_LABELS.get(class_id, f"class-{class_id}")


@dataclass
class PipelineOutput:
    """Tracks generated output files for a single inference run."""

    image_path: Path
    output_dir: Path
    files: Dict[str, Path]
    predictions: Dict[str, List[dict]]


def normalize_tasks(tasks: Iterable[str]) -> Set[str]:
    """Normalize requested tasks and expand 'all' token."""
    normalized = {task.strip().lower() for task in tasks if task.strip()}
    if "all" in normalized:
        return set(VALID_TASKS)
    invalid = sorted(normalized - VALID_TASKS)
    if invalid:
        raise ValueError(f"Invalid task(s): {invalid}. Valid options: {sorted(VALID_TASKS)} or 'all'.")
    return normalized


def _fdi_from_quadrant_and_class(quadrant_id: int, class_id: int) -> int:
    """Map mirrored 8-class tooth id and quadrant id to global FDI label."""
    if quadrant_id == 0:
        return 8 - class_id
    if quadrant_id == 1:
        return 9 + class_id
    if quadrant_id == 2:
        return 24 - class_id
    if quadrant_id == 3:
        return 25 + class_id
    raise ValueError(f"Unsupported quadrant id: {quadrant_id}")


def _teeth_from_quadrant_crops(crops_data: List[dict], model_registry: ModelRegistry, conf_teeth: float) -> List[dict]:
    """Run teeth segmentation on each quadrant crop and remap detections to global coordinates with FDI labels."""
    all_detections: List[dict] = []
    for item in crops_data:
        q_id = int(item["class_id"])
        px1, py1 = item["top_left"]
        crop_result = segment_teeth_on_array(item["crop"], model_registry=model_registry, conf=conf_teeth)
        if crop_result.boxes is None or len(crop_result.boxes) == 0:
            continue

        boxes = crop_result.boxes.xyxy.cpu().numpy().tolist()
        classes = crop_result.boxes.cls.cpu().numpy().astype(int).tolist()
        confs = crop_result.boxes.conf.cpu().numpy().tolist()
        masks = crop_result.masks.xy if crop_result.masks is not None else [None] * len(classes)

        for box, cls_id, conf, mask in zip(boxes, classes, confs, masks):
            fdi = _fdi_from_quadrant_and_class(q_id, int(cls_id))
            gx1, gy1, gx2, gy2 = [float(box[0] + px1), float(box[1] + py1), float(box[2] + px1), float(box[3] + py1)]
            global_mask = (
                [[float(x + px1), float(y + py1)] for x, y in mask]
                if mask is not None
                else None
            )
            all_detections.append(
                {
                    "quadrant_id": q_id,
                    "class_id": int(cls_id),
                    "label": f"FDI-{fdi}",
                    "fdi": int(fdi),
                    "confidence": float(conf),
                    "box_xyxy": [gx1, gy1, gx2, gy2],
                    "mask_xy": global_mask,
                }
            )
    return all_detections


def _periapical_from_quadrant_crops(crops_data: List[dict], model_registry: ModelRegistry, conf_periapical: float) -> List[dict]:
    """Run periapical detector on each quadrant crop and remap to global coordinates."""
    all_detections: List[dict] = []
    for item in crops_data:
        q_id = int(item["class_id"])
        px1, py1 = item["top_left"]
        crop_result = detect_periapical_on_array(item["crop"], model_registry=model_registry, conf=conf_periapical)
        if crop_result.boxes is None or len(crop_result.boxes) == 0:
            continue

        boxes = crop_result.boxes.xyxy.cpu().numpy().tolist()
        classes = crop_result.boxes.cls.cpu().numpy().astype(int).tolist()
        confs = crop_result.boxes.conf.cpu().numpy().tolist()
        masks = crop_result.masks.xy if crop_result.masks is not None else [None] * len(classes)

        for box, cls_id, conf, mask in zip(boxes, classes, confs, masks):
            gx1, gy1, gx2, gy2 = [float(box[0] + px1), float(box[1] + py1), float(box[2] + px1), float(box[3] + py1)]
            global_mask = (
                [[float(x + px1), float(y + py1)] for x, y in mask]
                if mask is not None
                else None
            )
            all_detections.append(
                {
                    "quadrant_id": q_id,
                    "class_id": int(cls_id),
                    "label": f"P{int(cls_id)}",
                    "confidence": float(conf),
                    "box_xyxy": [gx1, gy1, gx2, gy2],
                    "mask_xy": global_mask,
                }
            )
    return all_detections


def _teeth_classification_from_quadrant_crops(
    crops_data: List[dict],
    model_registry: ModelRegistry,
    conf_tc: float,
) -> List[dict]:
    """Run teeth pathology/classification model on each crop; remap masks/boxes to global coordinates."""
    all_detections: List[dict] = []
    for item in crops_data:
        q_id = int(item["class_id"])
        px1, py1 = item["top_left"]
        crop_result = classify_teeth_on_array(item["crop"], model_registry=model_registry, conf=conf_tc)
        model_names = getattr(crop_result, "names", None)
        if crop_result.boxes is None or len(crop_result.boxes) == 0:
            continue

        boxes = crop_result.boxes.xyxy.cpu().numpy().tolist()
        classes = crop_result.boxes.cls.cpu().numpy().astype(int).tolist()
        confs = crop_result.boxes.conf.cpu().numpy().tolist()
        masks = crop_result.masks.xy if crop_result.masks is not None else [None] * len(classes)

        for box, cls_id, conf, mask in zip(boxes, classes, confs, masks):
            cid = int(cls_id)
            gx1, gy1, gx2, gy2 = [float(box[0] + px1), float(box[1] + py1), float(box[2] + px1), float(box[3] + py1)]
            global_mask = (
                [[float(x + px1), float(y + py1)] for x, y in mask]
                if mask is not None
                else None
            )
            label = _resolve_teeth_class_label(cid, model_names)
            all_detections.append(
                {
                    "quadrant_id": q_id,
                    "class_id": cid,
                    "label": label,
                    "confidence": float(conf),
                    "box_xyxy": [gx1, gy1, gx2, gy2],
                    "mask_xy": global_mask,
                }
            )
    return all_detections


def _run_crop_models(
    crops_data: List[dict],
    registry: ModelRegistry,
    selected_tasks: Set[str],
    conf_teeth: float,
    conf_periapical: float,
    conf_teeth_classification: float,
    *,
    parallel: bool,
) -> Dict[str, List[dict]]:
    """Run crop-level teeth / periapical / classification models (parallel or sequential)."""
    if not crops_data:
        return {}

    if not parallel:
        out: Dict[str, List[dict]] = {}
        if "teeth" in selected_tasks:
            out["teeth"] = _teeth_from_quadrant_crops(crops_data, registry, conf_teeth)
        if "periapical" in selected_tasks:
            out["periapical"] = _periapical_from_quadrant_crops(crops_data, registry, conf_periapical)
        if "teeth_classification" in selected_tasks:
            out["teeth_classification"] = _teeth_classification_from_quadrant_crops(
                crops_data, registry, conf_teeth_classification
            )
        return out

    futures: Dict[str, Future] = {}
    with ThreadPoolExecutor(max_workers=3) as executor:
        if "teeth" in selected_tasks:
            futures["teeth"] = executor.submit(_teeth_from_quadrant_crops, crops_data, registry, conf_teeth)
        if "periapical" in selected_tasks:
            futures["periapical"] = executor.submit(_periapical_from_quadrant_crops, crops_data, registry, conf_periapical)
        if "teeth_classification" in selected_tasks:
            futures["teeth_classification"] = executor.submit(
                _teeth_classification_from_quadrant_crops,
                crops_data,
                registry,
                conf_teeth_classification,
            )
        return {name: fut.result() for name, fut in futures.items()}




def run_pipeline(
    image_path: ImagePath,
    tasks: Iterable[str],
    output_dir: ImagePath = "outputs",
    conf_quadrants: float = 0.3,
    conf_teeth: float = 0.3,
    conf_periapical: float = 0.3,
    conf_teeth_classification: float = 0.3,
    model_registry: Optional[ModelRegistry] = None,
    *,
    parallel_crop_models: bool = True,
) -> PipelineOutput:
    """Run selected tasks for one image and save visualized outputs.

    Stage 1: quadrant segmentation and crops. Stage 2: teeth segmentation,
    periapical on crops, teeth classification on crops — in parallel when
    ``parallel_crop_models`` is true (default), else sequentially (lower peak RAM).

    Args:
        image_path: Input panoramic image path.
        tasks: Task names or ``all``.
        output_dir: Directory for overlays and JSON artifacts.
        conf_quadrants: Confidence threshold for quadrant segmentation.
        conf_teeth: Confidence threshold for teeth segmentation on crops.
        conf_periapical: Confidence threshold for periapical detection on crops.
        conf_teeth_classification: Confidence threshold for teeth classification on crops.
        model_registry: Optional shared registry to reuse loaded weights across runs.
        parallel_crop_models: When false, run crop-stage models one after another.
    """
    image = to_path(image_path)
    if not image.exists():
        raise FileNotFoundError(f"Image not found: {image}")

    selected_tasks = normalize_tasks(tasks)
    out_dir = ensure_dir(output_dir)
    out_files: Dict[str, Path] = {}
    predictions: Dict[str, List[dict]] = {}
    base_name = image.stem

    registry = model_registry if model_registry is not None else ModelRegistry()
    original_bgr = read_image_bgr(image)
    processed_image = original_bgr
    crops_data: List[dict] = []

    needs_crops = bool(selected_tasks & {"quadrants", "teeth", "periapical", "teeth_classification"})
    if needs_crops:
        processed_image, crops_data, _ = get_quadrant_crops(
            image_path=image,
            model_registry=registry,
            padding=40,
            quadrant_conf=conf_quadrants,
        )

    if "quadrants" in selected_tasks:
        quadrants_overlay = draw_quadrants_overlay(processed_image=processed_image, crops_data=crops_data, alpha=0.35)
        quadrants_overlay_path = out_dir / f"{base_name}_quadrants_overlay.jpg"
        save_bgr(quadrants_overlay_path, quadrants_overlay)
        out_files["quadrants_overlay"] = quadrants_overlay_path

        quadrants_grid_path = out_dir / f"{base_name}_quadrants_grid.png"
        save_quadrant_grid(
            path=quadrants_grid_path,
            processed_image=processed_image,
            crops_data=crops_data,
            filename=image.name,
            alpha=0.35,
        )
        out_files["quadrants_grid"] = quadrants_grid_path
        predictions["quadrants"] = [
            {
                "quadrant_id": int(item["class_id"]),
                "box_xyxy": [float(v) for v in item["original_box"]],
                "top_left": [int(item["top_left"][0]), int(item["top_left"][1])],
            }
            for item in crops_data
        ]

    stage2 = _run_crop_models(
        crops_data,
        registry,
        selected_tasks,
        conf_teeth,
        conf_periapical,
        conf_teeth_classification,
        parallel=parallel_crop_models,
    )

    if "teeth" in selected_tasks:
        teeth_preds = stage2.get("teeth", [])
        predictions["teeth"] = teeth_preds
        teeth_overlay = draw_global_detections_overlay(base_image_bgr=processed_image, detections=teeth_preds, alpha=0.35)
        teeth_path = out_dir / f"{base_name}_teeth_overlay.jpg"
        save_bgr(teeth_path, teeth_overlay)
        out_files["teeth_overlay"] = teeth_path

    if "periapical" in selected_tasks:
        peri_preds = stage2.get("periapical", [])
        predictions["periapical"] = peri_preds
        peri_overlay = draw_global_detections_overlay(base_image_bgr=processed_image, detections=peri_preds, alpha=0.35)
        peri_path = out_dir / f"{base_name}_periapical_quadrants_overlay.jpg"
        save_bgr(peri_path, peri_overlay)
        out_files["periapical_quadrants_overlay"] = peri_path

    if "teeth_classification" in selected_tasks:
        tc_preds = stage2.get("teeth_classification", [])
        predictions["teeth_classification"] = tc_preds
        tc_overlay = draw_global_detections_overlay(base_image_bgr=processed_image, detections=tc_preds, alpha=0.35)
        tc_path = out_dir / f"{base_name}_teeth_classification_overlay.jpg"
        save_bgr(tc_path, tc_overlay)
        out_files["teeth_classification_overlay"] = tc_path

    original_copy_path = out_dir / f"{base_name}_original.jpg"
    save_bgr(original_copy_path, original_bgr)
    out_files["original"] = original_copy_path

    json_path = out_dir / f"{base_name}_predictions.json"
    json_payload = {
        "image_path": str(image),
        "tasks": sorted(selected_tasks),
        "predictions": predictions,
    }
    json_path.write_text(json.dumps(json_payload, indent=2), encoding="utf-8")
    out_files["predictions_json"] = json_path

    return PipelineOutput(image_path=image, output_dir=out_dir, files=out_files, predictions=predictions)
