"""Optional dev token issuance."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import jwt
from fastapi import APIRouter, HTTPException

from backend.app.core.config import get_settings
from backend.app.schemas.auth import TokenRequest, TokenResponse

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/token", response_model=TokenResponse)
async def issue_dev_token(body: TokenRequest) -> TokenResponse:
    """Return a short-lived JWT (only when ``auth_dev_login_enabled`` is true)."""
    settings = get_settings()
    if not settings.auth_dev_login_enabled:
        raise HTTPException(status_code=404, detail="Not found.")
    if not settings.auth_jwt_secret:
        raise HTTPException(
            status_code=503,
            detail="Server is not configured with TOOTHFAIRY_AUTH_JWT_SECRET.",
        )
    now = datetime.now(timezone.utc)
    exp = now + timedelta(minutes=settings.auth_access_token_ttl_minutes)
    token = jwt.encode(
        {"sub": body.user_id, "iat": int(now.timestamp()), "exp": exp},
        settings.auth_jwt_secret,
        algorithm=settings.auth_jwt_algorithm,
    )
    if isinstance(token, bytes):
        token = token.decode("ascii")
    return TokenResponse(access_token=token)
