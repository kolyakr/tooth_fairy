"""Shared FastAPI dependencies."""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass
from typing import Annotated

import jwt
from fastapi import Depends, Header, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.config import get_settings
from backend.app.core.database import get_db
from backend.app.core.guest_cookie import verify_guest_session_cookie
from backend.app.services.domain_errors import NotFoundError
from backend.app.services.guest_workspace import GuestWorkspace

DbSession = Annotated[AsyncSession, Depends(get_db)]


@dataclass(frozen=True)
class AuthPrincipal:
    """Authenticated user identity from JWT ``sub``."""

    user_id: str


async def get_optional_principal(
    authorization: Annotated[str | None, Header()] = None,
) -> AuthPrincipal | None:
    """Parse ``Authorization: Bearer`` JWT when secret is configured."""
    settings = get_settings()
    if not settings.auth_jwt_secret or not authorization:
        return None
    if not authorization.lower().startswith("bearer "):
        return None
    token = authorization[7:].strip()
    try:
        payload = jwt.decode(
            token,
            settings.auth_jwt_secret,
            algorithms=[settings.auth_jwt_algorithm],
        )
        sub = payload.get("sub")
        if not sub:
            return None
        return AuthPrincipal(user_id=str(sub))
    except jwt.PyJWTError:
        return None


async def require_principal(
    principal: Annotated[AuthPrincipal | None, Depends(get_optional_principal)],
) -> AuthPrincipal:
    """Require a valid Bearer JWT."""
    if principal is None:
        raise HTTPException(status_code=401, detail="Authentication required.")
    return principal


def get_guest_session_id(request: Request) -> uuid.UUID | None:
    """Return verified guest session id from cookie, if any."""
    settings = get_settings()
    raw = request.cookies.get(settings.guest_session_cookie_name)
    return verify_guest_session_cookie(settings, raw)


def get_guest_workspace(request: Request) -> GuestWorkspace:
    """Return process-global guest workspace."""
    return request.app.state.guest_workspace


def assert_guest_session_owns(request: Request, record_session_id: uuid.UUID) -> None:
    """Raise ``NotFoundError`` if guest cookie does not match analysis session."""
    sid = get_guest_session_id(request)
    if sid is None or sid != record_session_id:
        raise NotFoundError("Analysis not found.")


_model_registry_lock = threading.Lock()


async def get_model_registry(request: Request):
    """Return the shared Ultralytics registry, building it on first use."""
    existing = getattr(request.app.state, "model_registry", None)
    if existing is not None:
        return existing
    with _model_registry_lock:
        existing = getattr(request.app.state, "model_registry", None)
        if existing is not None:
            return existing
        from backend.app.core.model_registry import build_model_registry

        request.app.state.model_registry = build_model_registry()
    return request.app.state.model_registry
