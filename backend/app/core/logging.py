"""Structured logging setup for the API."""

from __future__ import annotations

import logging
import sys
from typing import Any


def configure_logging(level: int = logging.INFO) -> None:
    """Configure root logger with a concise format suitable for development."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    )
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger."""
    return logging.getLogger(name)


def log_extra(**kwargs: Any) -> dict[str, Any]:
    """Helper for structured extras (compatible with future JSON logging)."""
    return kwargs
