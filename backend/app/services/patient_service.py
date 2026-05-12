"""Patient lookups and creation."""

from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.db.models import Patient
from backend.app.schemas.patient import PatientCreate


async def get_or_create_patient(session: AsyncSession, payload: PatientCreate) -> Patient:
    """Return existing patient by ``patient_code`` or insert a new row."""
    stmt = select(Patient).where(Patient.patient_code == payload.patient_code)
    result = await session.execute(stmt)
    existing = result.scalar_one_or_none()
    if existing:
        return existing
    row = Patient(patient_code=payload.patient_code, name=payload.name, age=payload.age)
    session.add(row)
    await session.flush()
    return row


async def get_patient(session: AsyncSession, patient_id: uuid.UUID) -> Patient | None:
    """Load patient by primary key."""
    stmt = select(Patient).where(Patient.id == patient_id)
    result = await session.execute(stmt)
    return result.scalar_one_or_none()
