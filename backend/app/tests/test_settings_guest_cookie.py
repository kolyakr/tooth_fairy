"""Guest cookie defaults for HTTPS / cross-origin API consumers."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from backend.app.core.config import Settings


def test_guest_https_upgrades_default_lax_to_none() -> None:
    """Split UI (e.g. localhost) + HTTPS API needs SameSite=None for ``fetch`` credentials."""
    s = Settings(guest_cookie_secure=True, guest_cookie_samesite="lax")
    assert s.guest_cookie_samesite == "none"


def test_guest_http_keeps_lax() -> None:
    s = Settings(guest_cookie_secure=False, guest_cookie_samesite="lax")
    assert s.guest_cookie_samesite == "lax"


def test_guest_none_without_secure_raises() -> None:
    with pytest.raises(ValidationError):
        Settings(guest_cookie_secure=False, guest_cookie_samesite="none")


def test_guest_explicit_none_with_secure() -> None:
    s = Settings(guest_cookie_secure=True, guest_cookie_samesite="none")
    assert s.guest_cookie_samesite == "none"
