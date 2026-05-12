"""Patient-related API schemas."""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class PatientCreate(BaseModel):
    """Payload for registering or referencing a patient."""

    patient_code: str = Field(..., min_length=1, max_length=64)
    name: str = Field(..., min_length=1, max_length=255)
    age: int | None = Field(default=None, ge=0, le=130)


class PatientRead(BaseModel):
    """Patient returned by the API."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    patient_code: str
    name: str
    age: int | None
    created_at: datetime
