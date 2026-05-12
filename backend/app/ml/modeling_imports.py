"""Ensure the ``modeling/`` tree is importable as ``utils`` and ``modeling`` packages."""

from __future__ import annotations

import sys
from pathlib import Path

from backend.app.core.config import get_settings


def setup_modeling_path() -> Path:
    """Prepend the modeling project root to ``sys.path`` if needed.

    The research code lives under ``<repo>/modeling`` and imports ``from utils...``.

    Returns:
        Absolute path to the modeling root.
    """
    root = get_settings().resolved_modeling_root()
    s = str(root)
    if s not in sys.path:
        sys.path.insert(0, s)
    return root
