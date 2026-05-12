from __future__ import annotations

import matplotlib.pyplot as plt
from pathlib import Path

from .common import apply_clahe
from .model_inference import (
    ModelRegistry,
    get_quadrant_crops as _get_quadrant_crops,
    segment_quadrants as _segment_quadrants,
    segment_teeth as _segment_teeth,
)
from .visualization import draw_instances_overlay, save_quadrant_grid

_DEFAULT_REGISTRY = ModelRegistry()


def segment_quadrants(image_path, conf=0.5):
    """Backward-compatible wrapper with old signature."""
    return _segment_quadrants(image_path=image_path, model_registry=_DEFAULT_REGISTRY, conf=conf)


def get_quadrant_crops(image_path, padding=40):
    """Backward-compatible wrapper with old signature."""
    return _get_quadrant_crops(image_path=image_path, model_registry=_DEFAULT_REGISTRY, padding=padding)


def segment_teeth(image_path, conf=0.5):
    """Backward-compatible wrapper with old signature."""
    return _segment_teeth(image_path=image_path, model_registry=_DEFAULT_REGISTRY, conf=conf)


def visualize_quadrant_crops(processed_image, crops_data, filename, alpha=0.3):
    """Backward-compatible helper for notebook visualization."""
    # The new implementation writes a figure to disk internally; here we mimic old show() behavior.
    save_quadrant_grid(
        path=Path("_tmp_quadrants_grid.png"),
        processed_image=processed_image,
        crops_data=crops_data,
        filename=filename,
        alpha=alpha,
    )
    img = plt.imread("_tmp_quadrants_grid.png")
    plt.figure(figsize=(12, 10))
    plt.imshow(img)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def visualize_teeth(result, alpha=0.35):
    """Backward-compatible helper for notebook visualization."""
    final_img = draw_instances_overlay(result, alpha=alpha, label_prefix="T")
    plt.figure(figsize=(14, 12))
    plt.imshow(final_img[:, :, ::-1])
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def get_default_registry() -> ModelRegistry:
    """Return a default model registry for notebook workflows."""
    return ModelRegistry()