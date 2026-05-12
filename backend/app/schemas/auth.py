"""Auth token contracts."""

from __future__ import annotations

from pydantic import BaseModel, Field


class TokenRequest(BaseModel):
    """Dev-only login body."""

    user_id: str = Field(..., min_length=1, max_length=128)


class TokenResponse(BaseModel):
    """Bearer access token."""

    access_token: str
    token_type: str = "bearer"
