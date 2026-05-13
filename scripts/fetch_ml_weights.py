#!/usr/bin/env python3
"""Download YOLO ``.pt`` weights when binaries are not present in the working tree.

Repository ``.gitignore`` excludes ``*.pt``; use this script (or manual copies) to
place files under ``modeling/models/``.

Set any of the following to an HTTPS URL for the corresponding ``best.pt``:

- ``TOOTHFAIRY_FETCH_WEIGHT_QUADRANTS_URL``
- ``TOOTHFAIRY_FETCH_WEIGHT_TEETH_URL``
- ``TOOTHFAIRY_FETCH_WEIGHT_PERIAPICAL_URL``
- ``TOOTHFAIRY_FETCH_WEIGHT_TEETH_CLASSIFICATION_URL``

Run from the repository root::

    python scripts/fetch_ml_weights.py

You can chain the script after dependency install in a setup script; keep download
URLs in environment variables or a secret store if the artifacts are private.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent.parent

# Must match default paths in ``backend.app.core.config.Settings.resolved_model_paths``.
_TARGETS: list[tuple[str, Path]] = [
    ("TOOTHFAIRY_FETCH_WEIGHT_QUADRANTS_URL", ROOT / "modeling/models/quadrant segmentation/best.pt"),
    ("TOOTHFAIRY_FETCH_WEIGHT_TEETH_URL", ROOT / "modeling/models/teeth segmentation/best.pt"),
    ("TOOTHFAIRY_FETCH_WEIGHT_PERIAPICAL_URL", ROOT / "modeling/models/periapical detector (cropped)/best.pt"),
    ("TOOTHFAIRY_FETCH_WEIGHT_TEETH_CLASSIFICATION_URL", ROOT / "modeling/models/teeth classification/best.pt"),
]


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = Request(url, headers={"User-Agent": "ToothFairy-fetch-ml-weights/1"})
    with urlopen(req, timeout=600) as resp:
        data = resp.read()
    tmp = dest.with_suffix(dest.suffix + ".part")
    tmp.write_bytes(data)
    tmp.replace(dest)


def main() -> int:
    any_url = False
    for env_key, dest in _TARGETS:
        raw = os.environ.get(env_key, "")
        url = raw.strip() if raw else ""
        if not url:
            continue
        any_url = True
        print(f"{env_key}: -> {dest}", file=sys.stderr)
        try:
            _download(url, dest)
        except (HTTPError, URLError, OSError, TimeoutError) as exc:
            print(f"error: failed to download {env_key}: {exc}", file=sys.stderr)
            return 1
    if not any_url:
        print(
            "No TOOTHFAIRY_FETCH_WEIGHT_*_URL variables set; skipping downloads. "
            "Ensure ``.pt`` files exist on disk or set URLs for a deploy fetch.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
