"""CRUD for findings with mandatory audit side-effects."""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.db.models import (
    Analysis,
    AuditActionType,
    Finding,
    FindingLayer,
    FindingSource,
)
from backend.app.schemas.finding import FindingCreate, FindingRead, FindingUpdate, PolygonPoint
from backend.app.services.audit_service import append_system_audit, snapshot_finding
from backend.app.services.domain_errors import NotFoundError
from backend.app.services.guest_workspace import GuestFinding

def finding_to_read(row: Finding) -> FindingRead:
    """Map ORM row to API schema (``finding`` mirrors frontend naming)."""
    raw_poly = row.polygon or []
    poly: list[PolygonPoint] = []
    for p in raw_poly:
        if isinstance(p, dict) and "x" in p and "y" in p:
            poly.append(PolygonPoint(x=float(p["x"]), y=float(p["y"])))
        elif isinstance(p, (list, tuple)) and len(p) >= 2:
            poly.append(PolygonPoint(x=float(p[0]), y=float(p[1])))
    return FindingRead(
        id=row.id,
        tooth_label=row.tooth_label,
        finding=row.finding_class,
        confidence=row.confidence,
        accepted=row.accepted,
        polygon=poly,
        layer=row.layer,
        source=row.source,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


def guest_finding_to_read(row: GuestFinding) -> FindingRead:
    """Map in-memory guest finding to API schema."""
    raw_poly = row.polygon or []
    poly: list[PolygonPoint] = []
    for p in raw_poly:
        if isinstance(p, dict) and "x" in p and "y" in p:
            poly.append(PolygonPoint(x=float(p["x"]), y=float(p["y"])))
    return FindingRead(
        id=row.id,
        tooth_label=row.tooth_label,
        finding=row.finding_class,
        confidence=row.confidence,
        accepted=row.accepted,
        polygon=poly,
        layer=row.layer,
        source=row.source,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


async def list_findings(session: AsyncSession, analysis_id: uuid.UUID) -> list[Finding]:
    """Return all findings for an analysis."""
    stmt = select(Finding).where(Finding.analysis_id == analysis_id).order_by(Finding.created_at)
    result = await session.execute(stmt)
    return list(result.scalars().all())


async def get_finding(session: AsyncSession, finding_id: uuid.UUID) -> Finding:
    """Load a finding or raise."""
    stmt = select(Finding).where(Finding.id == finding_id)
    result = await session.execute(stmt)
    row = result.scalar_one_or_none()
    if row is None:
        raise NotFoundError("Finding not found.")
    return row


async def create_manual_finding(
    session: AsyncSession,
    *,
    analysis_id: uuid.UUID,
    payload: FindingCreate,
    reviewer: str,
) -> Finding:
    """Persist a dentist-drawn finding."""
    stmt = select(Analysis).where(Analysis.id == analysis_id)
    res = await session.execute(stmt)
    if res.scalar_one_or_none() is None:
        raise NotFoundError("Analysis not found.")

    layer = payload.layer
    poly_json: list[dict[str, float]] = [p.model_dump() for p in payload.polygon]
    row = Finding(
        analysis_id=analysis_id,
        tooth_label=payload.tooth_label,
        finding_class=payload.finding,
        layer=layer,
        confidence=payload.confidence,
        polygon=poly_json,
        accepted=payload.accepted,
        source=payload.source,
    )
    session.add(row)
    await session.flush()
    snap = finding_to_read(row)
    await append_system_audit(
        session,
        analysis_id=analysis_id,
        reviewer=reviewer,
        action=f"Created manual finding {row.id}",
        action_type=AuditActionType.CREATE,
        target_finding_id=row.id,
        after=snap.model_dump(mode="json"),
    )
    return row


async def update_finding(
    session: AsyncSession,
    *,
    finding_id: uuid.UUID,
    payload: FindingUpdate,
    reviewer: str,
) -> Finding:
    """Apply a partial update with audit trail."""
    row = await get_finding(session, finding_id)
    before = await snapshot_finding(session, finding_id)
    if payload.tooth_label is not None:
        row.tooth_label = payload.tooth_label
    if payload.finding is not None:
        row.finding_class = payload.finding
    if payload.confidence is not None:
        row.confidence = payload.confidence
    if payload.accepted is not None:
        row.accepted = payload.accepted
    if payload.polygon is not None:
        row.polygon = [p.model_dump() for p in payload.polygon]
    if payload.layer is not None:
        row.layer = payload.layer
    await session.flush()
    after = await snapshot_finding(session, finding_id)
    await append_system_audit(
        session,
        analysis_id=row.analysis_id,
        reviewer=reviewer,
        action=f"Updated finding {finding_id}",
        action_type=AuditActionType.UPDATE,
        target_finding_id=finding_id,
        before=before,
        after=after,
    )
    return row


async def delete_finding(session: AsyncSession, *, finding_id: uuid.UUID, reviewer: str) -> None:
    """Delete a finding with audit."""
    row = await get_finding(session, finding_id)
    before = await snapshot_finding(session, finding_id)
    analysis_id = row.analysis_id
    await append_system_audit(
        session,
        analysis_id=analysis_id,
        reviewer=reviewer,
        action=f"Deleted finding {finding_id}",
        action_type=AuditActionType.DELETE,
        target_finding_id=finding_id,
        before=before,
        after=None,
    )
    await session.execute(delete(Finding).where(Finding.id == finding_id))
