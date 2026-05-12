from __future__ import annotations

from pathlib import Path
from typing import Union

import cv2
import numpy as np

ImagePath = Union[str, Path]


def to_path(path_like: ImagePath) -> Path:
    """Normalize a path-like object to an absolute pathlib.Path."""
    return Path(path_like).expanduser().resolve()


def read_image_bgr(image_path: ImagePath) -> np.ndarray:
    """Read an image as BGR and raise if file is invalid."""
    path = to_path(image_path)
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def apply_clahe(image_path: ImagePath, clip_limit: float = 3.0, tile_size: int = 8) -> np.ndarray:
    """Apply CLAHE to an input image and return grayscale output."""
    path = to_path(image_path)
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    return clahe.apply(gray)


def ensure_dir(path_like: ImagePath) -> Path:
    """Create a directory if it does not exist and return its absolute path."""
    path = to_path(path_like)
    path.mkdir(parents=True, exist_ok=True)
    return path
