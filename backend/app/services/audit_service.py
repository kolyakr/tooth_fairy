"""Persistence helpers for audit entries."""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.db.models import AuditActionType, AuditEntry, Finding
from backend.app.schemas.audit import AuditEntryCreate


async def list_audit_entries(session: AsyncSession, analysis_id: uuid.UUID) -> list[AuditEntry]:
    """Return audit rows for an analysis ordered newest first."""
    stmt = (
        select(AuditEntry)
        .where(AuditEntry.analysis_id == analysis_id)
        .order_by(AuditEntry.timestamp.desc())
    )
    result = await session.execute(stmt)
    return list(result.scalars().all())


async def append_audit_entry(
    session: AsyncSession,
    *,
    analysis_id: uuid.UUID,
    payload: AuditEntryCreate,
    target_finding_id: uuid.UUID | None = None,
) -> AuditEntry:
    """Insert an audit record."""
    entry = AuditEntry(
        analysis_id=analysis_id,
        reviewer=payload.reviewer,
        action=payload.action,
        action_type=payload.action_type,
        target_finding_id=target_finding_id or payload.target_id,
        before=payload.before,
        after=payload.after,
    )
    session.add(entry)
    await session.flush()
    return entry


async def append_system_audit(
    session: AsyncSession,
    *,
    analysis_id: uuid.UUID,
    reviewer: str,
    action: str,
    action_type: AuditActionType = AuditActionType.SYSTEM,
    target_finding_id: uuid.UUID | None = None,
    before: dict[str, Any] | None = None,
    after: dict[str, Any] | None = None,
) -> AuditEntry:
    """Convenience wrapper for server-generated audit rows."""
    payload = AuditEntryCreate(
        reviewer=reviewer,
        action=action,
        action_type=action_type,
        target_id=target_finding_id,
        before=before,
        after=after,
    )
    return await append_audit_entry(session, analysis_id=analysis_id, payload=payload)


async def snapshot_finding(session: AsyncSession, finding_id: uuid.UUID) -> dict[str, Any] | None:
    """Serialize a finding into an audit-friendly dict."""
    stmt = select(Finding).where(Finding.id == finding_id)
    result = await session.execute(stmt)
    row = result.scalar_one_or_none()
    if row is None:
        return None
    return {
        "id": str(row.id),
        "tooth_label": row.tooth_label,
        "finding": row.finding_class,
        "confidence": row.confidence,
        "accepted": row.accepted,
        "polygon": row.polygon or [],
        "layer": row.layer.value,
        "source": row.source.value,
    }
