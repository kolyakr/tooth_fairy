"""Audit log schemas mirroring the interactive viewer contract."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from backend.app.db.models import AuditActionType


class AuditEntryCreate(BaseModel):
    """Append-only audit row from client or server."""

    reviewer: str = Field(..., min_length=1)
    action: str
    action_type: AuditActionType = AuditActionType.SYSTEM
    target_id: uuid.UUID | None = Field(default=None, description="Finding id when applicable.")
    before: dict[str, Any] | None = None
    after: dict[str, Any] | None = None


class AuditEntryRead(BaseModel):
    """Audit row returned to clients."""

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    id: uuid.UUID
    reviewer: str
    action: str
    action_type: AuditActionType
    target_finding_id: uuid.UUID | None = Field(default=None, serialization_alias="target_id")
    before: dict[str, Any] | None = None
    after: dict[str, Any] | None = None
    timestamp: datetime


class CompleteReviewRequest(BaseModel):
    """Request body for dentist review completion."""

    reviewer: str = Field(..., min_length=1)
