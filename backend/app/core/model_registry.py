"""Singleton registry for YOLO weights (shared across inference runs)."""

from __future__ import annotations

from backend.app.core.config import get_settings
from backend.app.ml.modeling_imports import setup_modeling_path


def build_model_registry():
    """Construct a ``ModelRegistry`` with paths resolved from settings.

    Returns:
        Lazy-loading Ultralytics registry backed by absolute ``.pt`` paths.
    """
    setup_modeling_path()
    from utils.model_inference import ModelRegistry

    paths = get_settings().resolved_model_paths()
    return ModelRegistry(model_paths=paths)
