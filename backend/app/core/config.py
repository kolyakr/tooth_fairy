"""Application settings loaded from environment variables."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import List, Literal, Optional, Union

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _default_repo_root() -> Path:
    """Resolve repository root (parent of ``backend/``)."""
    # backend/app/core/config.py -> parents[3] == repo root
    return Path(__file__).resolve().parent.parent.parent.parent


class Settings(BaseSettings):
    """Runtime configuration for the API."""

    model_config = SettingsConfigDict(env_prefix="TOOTHFAIRY_", env_file=".env", extra="ignore")

    project_name: str = "ToothFairy API"
    api_v1_prefix: str = "/api/v1"
    debug: bool = False

    cors_origins: List[str] = Field(default_factory=lambda: ["http://localhost:3000"])

    database_url: str = Field(
        default="postgresql+asyncpg://toothfairy:toothfairy@localhost:5432/toothfairy",
        description="SQLAlchemy async URL for the primary database.",
    )

    auth_jwt_secret: str | None = Field(
        default=None,
        description="HS256 secret for Bearer JWTs; when unset, no token is accepted as authenticated.",
    )
    auth_jwt_algorithm: str = Field(default="HS256", description="JWT algorithm for access tokens.")
    auth_access_token_ttl_minutes: int = Field(default=60, ge=5, le=24 * 60)
    auth_dev_login_enabled: bool = Field(
        default=False,
        description="If true, expose POST /api/v1/auth/token for issuing test JWTs (never enable in production).",
    )

    guest_session_cookie_name: str = Field(default="toothfairy_guest")
    guest_session_ttl_seconds: int = Field(default=86400, ge=300, le=604800)
    guest_max_analyses_per_session: int = Field(default=50, ge=1, le=500)
    guest_cookie_secure: bool = Field(
        default=False,
        description="Set True behind HTTPS so the guest session cookie uses the Secure flag.",
    )
    guest_cookie_samesite: Literal["lax", "strict", "none"] = Field(
        default="lax",
        description=(
            "SameSite policy for the guest cookie. When guest_cookie_secure is True, "
            "the default lax value is upgraded to none so browsers attach the cookie on "
            "cross-origin fetch (localhost UI → HTTPS API)."
        ),
    )

    repo_root: Path = Field(default_factory=_default_repo_root)
    modeling_root: Optional[Path] = Field(
        default=None,
        description="Directory containing ``utils`` and ``models`` for inference; defaults to ``<repo>/modeling``.",
    )

    model_quadrants_path: Optional[Path] = None
    model_teeth_path: Optional[Path] = None
    model_periapical_path: Optional[Path] = None
    model_teeth_classification_path: Optional[Path] = None

    conf_quadrants: float = 0.3
    conf_teeth: float = 0.3
    conf_periapical: float = 0.3
    conf_teeth_classification: float = 0.3

    @field_validator("guest_cookie_samesite", mode="before")
    @classmethod
    def _normalize_guest_samesite(cls, v: object) -> object:
        if isinstance(v, str):
            return v.strip().lower()
        return v

    @model_validator(mode="after")
    def _guest_cookie_samesite_for_https(self) -> Settings:
        """Align guest cookie policy with how browsers treat cross-origin ``fetch``.

        With ``Secure`` (HTTPS), ``SameSite=Lax`` cookies are not attached to cross-site
        requests (e.g. Next.js on ``http://localhost:3000`` calling this API on another
        host). Guest analyses then 404 on poll because the session cookie is missing.
        Default upgrade: ``Lax`` + ``Secure`` → ``None`` so ``credentials: \"include\"``
        works for typical split UI/API deployments.
        """
        if self.guest_cookie_samesite == "none" and not self.guest_cookie_secure:
            raise ValueError(
                "TOOTHFAIRY_GUEST_COOKIE_SAMESITE=none requires TOOTHFAIRY_GUEST_COOKIE_SECURE=true "
                "(browsers reject SameSite=None without Secure)."
            )
        if self.guest_cookie_secure and self.guest_cookie_samesite == "lax":
            return self.model_copy(update={"guest_cookie_samesite": "none"})
        return self

    @field_validator(
        "repo_root",
        "modeling_root",
        "model_quadrants_path",
        "model_teeth_path",
        "model_periapical_path",
        "model_teeth_classification_path",
        mode="before",
    )
    @classmethod
    def _expand_path(cls, v: Optional[Union[Path, str]]) -> Optional[Path]:
        if v is None:
            return None
        return Path(v).expanduser().resolve()

    def resolved_modeling_root(self) -> Path:
        """Return absolute path to the modeling package root."""
        return self.modeling_root or (self.repo_root / "modeling")

    def resolved_model_paths(self) -> dict[str, str]:
        """Absolute paths for YOLO weights keyed like ``model_inference.DEFAULT_MODEL_PATHS``."""
        root = self.resolved_modeling_root()
        mq = self.model_quadrants_path or root / "models" / "quadrant segmentation" / "best.pt"
        mt = self.model_teeth_path or root / "models" / "teeth segmentation" / "best.pt"
        mp = self.model_periapical_path or root / "models" / "periapical detector (cropped)" / "best.pt"
        mtc = self.model_teeth_classification_path or root / "models" / "teeth classification" / "best.pt"
        return {
            "quadrants": str(mq),
            "teeth": str(mt),
            "periapical": str(mp),
            "teeth_classification": str(mtc),
        }


@lru_cache
def get_settings() -> Settings:
    """Return cached settings singleton."""
    return Settings()
