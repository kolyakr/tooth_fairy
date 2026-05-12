"""Guest (non-persisted) analysis lifecycle: upload metadata, detail, review completion."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import BinaryIO

from backend.app.db.models import AnalysisStatus, AuditActionType, FindingSource, ImageAssetKind
from backend.app.schemas.analysis import AnalysisDetail
from backend.app.schemas.finding import FindingCreate, FindingUpdate
from backend.app.services.analysis_service import parse_scan_datetime
from backend.app.services.domain_errors import ValidationDomainError
from backend.app.services.guest_workspace import GuestAnalysisRecord, GuestFinding, GuestImageAsset, GuestWorkspace


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


async def create_guest_analysis_for_upload(
    workspace: GuestWorkspace,
    *,
    session_id: uuid.UUID,
    patient_code: str,
    patient_name: str,
    age: int | None,
    scan_date_raw: str | None,
    chief_complaint: str | None,
    filename: str,
    file_obj: BinaryIO,
    file_size_limit: int = 80 * 1024 * 1024,
) -> GuestAnalysisRecord:
    """Validate upload and register an in-memory analysis (no DB)."""
    data = file_obj.read()
    if len(data) > file_size_limit:
        raise ValidationDomainError("Uploaded file exceeds size limit.")

    from backend.app.image_utils import mime_and_dimensions

    mime, width, height = mime_and_dimensions(data)
    if mime not in {"image/jpeg", "image/png"}:
        raise ValidationDomainError("Only JPEG or PNG uploads are supported.")

    scan_dt = parse_scan_datetime(scan_date_raw)
    aid = uuid.uuid4()
    now = _utcnow()
    original = GuestImageAsset(
        kind=ImageAssetKind.ORIGINAL,
        mime_type=mime,
        data=data,
        width=width,
        height=height,
    )
    record = GuestAnalysisRecord(
        analysis_id=aid,
        session_id=session_id,
        patient_code=patient_code.strip(),
        patient_name=patient_name.strip(),
        patient_age=age,
        filename=filename,
        scan_date=scan_dt,
        chief_complaint=chief_complaint,
        status=AnalysisStatus.PENDING_AI,
        alert_level=None,
        error=None,
        reviewer=None,
        created_at=now,
        completed_at=None,
        assets={ImageAssetKind.ORIGINAL: original},
        findings=[],
        audit=[],
        report_pdf=None,
    )
    workspace.register_analysis(session_id=session_id, record=record)
    return record


def guest_analysis_to_detail(record: GuestAnalysisRecord) -> AnalysisDetail:
    """Map guest record to ``AnalysisDetail`` schema."""
    kinds = sorted(record.assets.keys(), key=lambda k: k.value)
    synthetic_patient_id = uuid.uuid5(uuid.NAMESPACE_URL, f"guest:{record.analysis_id}")
    return AnalysisDetail(
        id=record.analysis_id,
        patient_id=synthetic_patient_id,
        filename=record.filename,
        scan_date=record.scan_date,
        chief_complaint=record.chief_complaint,
        status=record.status,
        alert_level=record.alert_level,
        reviewer=record.reviewer,
        error=record.error,
        created_at=record.created_at,
        completed_at=record.completed_at,
        findings_count=len(record.findings),
        image_kinds=[k.value for k in kinds],
    )


def complete_guest_review(workspace: GuestWorkspace, *, analysis_id: uuid.UUID, reviewer: str) -> GuestAnalysisRecord:
    """Mark guest analysis reviewed (in-memory)."""
    record = workspace.require(analysis_id)
    rev = reviewer.strip()
    if not rev:
        raise ValidationDomainError("Reviewer identity is required.")
    accepted_n = sum(1 for f in record.findings if f.accepted and f.source in {FindingSource.AI, FindingSource.MANUAL})
    if accepted_n == 0:
        raise ValidationDomainError("At least one accepted finding is required.")
    record.reviewer = rev
    record.status = AnalysisStatus.REVIEWED
    workspace.append_audit(
        record,
        reviewer=rev,
        action="Marked case as review complete",
        action_type=AuditActionType.REVIEW,
    )
    return record


def _snapshot_guest_finding(f: GuestFinding) -> dict:
    return {
        "id": str(f.id),
        "tooth_label": f.tooth_label,
        "finding": f.finding_class,
        "confidence": f.confidence,
        "accepted": f.accepted,
        "polygon": f.polygon or [],
        "layer": f.layer.value,
        "source": f.source.value,
    }


def create_manual_guest_finding(
    workspace: GuestWorkspace,
    record: GuestAnalysisRecord,
    *,
    payload: FindingCreate,
    reviewer: str,
) -> GuestFinding:
    """Add a manual finding to a guest analysis."""
    ts = _utcnow()
    poly_json = [p.model_dump() for p in payload.polygon]
    row = GuestFinding(
        id=uuid.uuid4(),
        analysis_id=record.analysis_id,
        tooth_label=payload.tooth_label,
        finding_class=payload.finding,
        layer=payload.layer,
        confidence=payload.confidence,
        box_xyxy=None,
        polygon=poly_json,
        accepted=payload.accepted,
        source=payload.source,
        created_at=ts,
        updated_at=ts,
    )
    workspace.append_finding(record, row)
    from backend.app.services.finding_service import guest_finding_to_read

    snap = guest_finding_to_read(row)
    workspace.append_audit(
        record,
        reviewer=reviewer,
        action=f"Created manual finding {row.id}",
        action_type=AuditActionType.CREATE,
        target_finding_id=row.id,
        after=snap.model_dump(mode="json"),
    )
    return row


def update_guest_finding(
    workspace: GuestWorkspace,
    *,
    finding_id: uuid.UUID,
    payload: FindingUpdate,
    reviewer: str,
) -> GuestFinding:
    """Apply partial update to a guest finding."""
    pair = workspace.get_analysis_for_finding(finding_id)
    if pair is None:
        from backend.app.services.domain_errors import NotFoundError

        raise NotFoundError("Finding not found.")
    record, row = pair
    before = _snapshot_guest_finding(row)
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
    row.updated_at = _utcnow()
    after = _snapshot_guest_finding(row)
    workspace.append_audit(
        record,
        reviewer=reviewer,
        action=f"Updated finding {finding_id}",
        action_type=AuditActionType.UPDATE,
        target_finding_id=finding_id,
        before=before,
        after=after,
    )
    return row


def delete_guest_finding(workspace: GuestWorkspace, *, finding_id: uuid.UUID, reviewer: str) -> None:
    """Remove a guest finding with audit."""
    pair = workspace.get_analysis_for_finding(finding_id)
    if pair is None:
        from backend.app.services.domain_errors import NotFoundError

        raise NotFoundError("Finding not found.")
    record, row = pair
    before = _snapshot_guest_finding(row)
    workspace.remove_finding(record, finding_id)
    workspace.append_audit(
        record,
        reviewer=reviewer,
        action=f"Deleted finding {finding_id}",
        action_type=AuditActionType.DELETE,
        target_finding_id=finding_id,
        before=before,
        after=None,
    )
