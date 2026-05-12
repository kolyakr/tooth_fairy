"""Signed opaque guest session cookie (HMAC-SHA256, no third-party dependency)."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import uuid
from typing import Any

from backend.app.core.config import Settings


def _b64encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _b64decode(raw: str) -> bytes:
    pad = "=" * (-len(raw) % 4)
    return base64.urlsafe_b64decode(raw + pad)


def sign_guest_session_id(settings: Settings, session_id: uuid.UUID) -> str:
    """Return ``payload.sig`` cookie value encoding ``session_id``."""
    secret = (settings.auth_jwt_secret or "toothfairy-local-only-guest-hmac").encode("utf-8")
    payload_obj: dict[str, Any] = {"sid": str(session_id)}
    payload = _b64encode(json.dumps(payload_obj, separators=(",", ":")).encode("utf-8"))
    sig = hmac.new(secret, payload.encode("ascii"), hashlib.sha256).hexdigest()
    return f"{payload}.{sig}"


def verify_guest_session_cookie(settings: Settings, cookie_value: str | None) -> uuid.UUID | None:
    """Verify cookie and return session id, or ``None`` if invalid."""
    if not cookie_value or "." not in cookie_value:
        return None
    payload, sep, sig = cookie_value.rpartition(".")
    if sep != "." or not payload or not sig:
        return None
    secret = (settings.auth_jwt_secret or "toothfairy-local-only-guest-hmac").encode("utf-8")
    expected = hmac.new(secret, payload.encode("ascii"), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, sig):
        return None
    try:
        raw = _b64decode(payload)
        obj = json.loads(raw.decode("utf-8"))
        sid = obj.get("sid")
        if not isinstance(sid, str):
            return None
        return uuid.UUID(sid)
    except (ValueError, json.JSONDecodeError, KeyError):
        return None
