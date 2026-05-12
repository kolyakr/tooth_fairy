"""Finding schemas aligned with ``frontend/src/lib/mock-data.ts``."""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from backend.app.db.models import FindingLayer, FindingSource


class PolygonPoint(BaseModel):
    """Single vertex in image (scene) coordinates."""

    x: float
    y: float


class FindingBase(BaseModel):
    """Shared fields for findings."""

    tooth_label: str = Field(..., max_length=64)
    finding: str = Field(..., max_length=255, description="Clinical label / class name.")
    confidence: float = Field(..., ge=0.0, le=1.0)
    accepted: bool = True
    polygon: list[PolygonPoint] = Field(default_factory=list)
    layer: FindingLayer


class FindingCreate(FindingBase):
    """Manual finding creation."""

    source: FindingSource = FindingSource.MANUAL


class FindingUpdate(BaseModel):
    """Partial update for accept/reject, reclassify, geometry."""

    tooth_label: str | None = Field(default=None, max_length=64)
    finding: str | None = Field(default=None, max_length=255)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    accepted: bool | None = None
    polygon: list[PolygonPoint] | None = None
    layer: FindingLayer | None = None


class FindingRead(BaseModel):
    """Finding returned to the viewer — ``finding`` mirrors frontend naming."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    tooth_label: str
    finding: str
    confidence: float
    accepted: bool
    polygon: list[PolygonPoint]
    layer: FindingLayer
    source: FindingSource
    created_at: datetime
    updated_at: datetime
