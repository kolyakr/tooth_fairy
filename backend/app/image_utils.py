"""Image decoding helpers without pulling in the full inference stack."""

from __future__ import annotations

import io

from PIL import Image


def mime_and_dimensions(data: bytes) -> tuple[str, int, int]:
    """Return ``(mime_type, width, height)`` for JPEG or PNG bytes.

    Raises:
        ValueError: If bytes cannot be decoded as an image.
    """
    img = Image.open(io.BytesIO(data))
    img.load()
    w, h = img.size
    fmt = (img.format or "").upper()
    if fmt == "PNG":
        return "image/png", w, h
    if fmt in ("JPEG", "JPG", "MPO"):
        return "image/jpeg", w, h
    raise ValueError(f"Unsupported image format: {fmt}")
