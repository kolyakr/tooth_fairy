"""Analysis lifecycle: upload, listing, detail, and review completion."""

from __future__ import annotations

import uuid
from datetime import date, datetime, timezone
from typing import BinaryIO

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from backend.app.db.models import Analysis, AnalysisStatus, AuditActionType, Finding, ImageAssetKind, Patient
from backend.app.schemas.analysis import AnalysisDetail, AnalysisListItem
from backend.app.schemas.patient import PatientCreate
from backend.app.services.audit_service import append_system_audit
from backend.app.services.domain_errors import NotFoundError, ValidationDomainError
from backend.app.services.patient_service import get_or_create_patient


def parse_scan_datetime(raw: str | None) -> datetime | None:
    """Parse ISO date or datetime strings from multipart forms."""
    if raw is None or raw.strip() == "":
        return None
    text = raw.strip()
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        pass
    try:
        d = date.fromisoformat(text)
        return datetime(d.year, d.month, d.day, tzinfo=timezone.utc)
    except ValueError:
        return None


async def create_analysis_for_upload(
    session: AsyncSession,
    *,
    patient_code: str,
    patient_name: str,
    age: int | None,
    scan_date_raw: str | None,
    chief_complaint: str | None,
    filename: str,
    file_obj: BinaryIO,
    file_size_limit: int = 80 * 1024 * 1024,
) -> Analysis:
    """Persist patient (upsert), analysis row, and original image bytes."""
    payload = PatientCreate(patient_code=patient_code, name=patient_name, age=age)
    patient = await get_or_create_patient(session, payload)

    data = file_obj.read()
    if len(data) > file_size_limit:
        raise ValidationDomainError("Uploaded file exceeds size limit.")

    from backend.app.image_utils import mime_and_dimensions

    mime, width, height = mime_and_dimensions(data)
    if mime not in {"image/jpeg", "image/png"}:
        raise ValidationDomainError("Only JPEG or PNG uploads are supported.")

    scan_dt = parse_scan_datetime(scan_date_raw)

    analysis = Analysis(
        patient_id=patient.id,
        filename=filename,
        scan_date=scan_dt,
        chief_complaint=chief_complaint,
        status=AnalysisStatus.PENDING_AI,
    )
    session.add(analysis)
    await session.flush()

    from backend.app.db.models import ImageAsset, ImageAssetKind

    asset = ImageAsset(
        analysis_id=analysis.id,
        kind=ImageAssetKind.ORIGINAL,
        mime_type=mime,
        data=data,
        width=width,
        height=height,
    )
    session.add(asset)
    await session.commit()
    await session.refresh(analysis)
    return analysis


async def list_recent_analyses(session: AsyncSession, limit: int = 100) -> list[AnalysisListItem]:
    """Dashboard rows with patient metadata."""
    stmt = (
        select(Analysis, Patient)
        .join(Patient, Patient.id == Analysis.patient_id)
        .order_by(Analysis.created_at.desc())
        .limit(limit)
    )
    result = await session.execute(stmt)
    rows = result.all()
    out: list[AnalysisListItem] = []
    for analysis, patient in rows:
        out.append(
            AnalysisListItem(
                id=analysis.id,
                patient_name=patient.name,
                patient_id=patient.patient_code,
                scan_date=analysis.scan_date,
                status=analysis.status,
                alert_level=analysis.alert_level,
            )
        )
    return out


async def get_analysis(session: AsyncSession, analysis_id: uuid.UUID) -> Analysis:
    """Return analysis or raise ``NotFoundError``."""
    stmt = select(Analysis).where(Analysis.id == analysis_id)
    result = await session.execute(stmt)
    row = result.scalar_one_or_none()
    if row is None:
        raise NotFoundError("Analysis not found.")
    return row


async def get_analysis_detail(session: AsyncSession, analysis_id: uuid.UUID) -> AnalysisDetail:
    """Analysis plus aggregate fields for polling endpoints."""
    stmt = (
        select(Analysis)
        .where(Analysis.id == analysis_id)
        .options(selectinload(Analysis.image_assets))
    )
    result = await session.execute(stmt)
    analysis = result.scalar_one_or_none()
    if analysis is None:
        raise NotFoundError("Analysis not found.")

    fc_stmt = select(func.count()).select_from(Finding).where(Finding.analysis_id == analysis_id)
    fc = int((await session.execute(fc_stmt)).scalar_one())

    kinds = sorted({a.kind.value for a in analysis.image_assets})

    return AnalysisDetail(
        id=analysis.id,
        patient_id=analysis.patient_id,
        filename=analysis.filename,
        scan_date=analysis.scan_date,
        chief_complaint=analysis.chief_complaint,
        status=analysis.status,
        alert_level=analysis.alert_level,
        reviewer=analysis.reviewer,
        error=analysis.error,
        created_at=analysis.created_at,
        completed_at=analysis.completed_at,
        findings_count=fc,
        image_kinds=kinds,
    )


async def complete_review(
    session: AsyncSession,
    *,
    analysis_id: uuid.UUID,
    reviewer: str,
) -> Analysis:
    """Mark analysis reviewed after HITL checks."""
    analysis = await get_analysis(session, analysis_id)
    rev = reviewer.strip()
    if not rev:
        raise ValidationDomainError("Reviewer identity is required.")

    stmt = select(func.count()).select_from(Finding).where(
        Finding.analysis_id == analysis_id,
        Finding.accepted.is_(True),
    )
    accepted_n = int((await session.execute(stmt)).scalar_one())
    if accepted_n == 0:
        raise ValidationDomainError("At least one accepted finding is required.")

    analysis.reviewer = rev
    analysis.status = AnalysisStatus.REVIEWED
    await append_system_audit(
        session,
        analysis_id=analysis_id,
        reviewer=rev,
        action="Marked case as review complete",
        action_type=AuditActionType.REVIEW,
    )
    await session.commit()
    await session.refresh(analysis)
    return analysis
