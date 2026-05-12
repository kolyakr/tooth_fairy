"""Patient endpoints."""

from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends

from backend.app.api.deps import DbSession, require_principal
from backend.app.schemas.patient import PatientCreate, PatientRead
from backend.app.services.domain_errors import NotFoundError
from backend.app.services.patient_service import get_or_create_patient, get_patient

router = APIRouter(prefix="/patients", tags=["patients"])


@router.post("", response_model=PatientRead)
async def create_patient(
    session: DbSession,
    _: Annotated[object, Depends(require_principal)],
    payload: PatientCreate,
) -> PatientRead:
    """Create or return existing patient when codes collide."""
    patient = await get_or_create_patient(session, payload)
    await session.commit()
    await session.refresh(patient)
    return PatientRead.model_validate(patient)


@router.get("/{patient_id}", response_model=PatientRead)
async def read_patient(
    session: DbSession,
    _: Annotated[object, Depends(require_principal)],
    patient_id: uuid.UUID,
) -> PatientRead:
    """Fetch patient by id."""
    patient = await get_patient(session, patient_id)
    if patient is None:
        raise NotFoundError("Patient not found.")
    return PatientRead.model_validate(patient)
