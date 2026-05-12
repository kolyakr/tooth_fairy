"""Analysis upload, polling, images, audit, and review completion."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Annotated

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    Header,
    Query,
    Request,
    Response,
    UploadFile,
)
from fastapi.responses import JSONResponse
from sqlalchemy import desc, select

from backend.app.api.deps import (
    AuthPrincipal,
    DbSession,
    assert_guest_session_owns,
    get_guest_session_id,
    get_guest_workspace,
    get_model_registry,
    get_optional_principal,
)
from backend.app.core.config import get_settings
from backend.app.core.guest_cookie import sign_guest_session_id
from backend.app.db.models import AnalysisStatus, AuditActionType, ImageAssetKind, ReportAsset
from backend.app.db.models import ImageAsset as ImageAssetModel
from backend.app.schemas.analysis import (
    AnalysisCreateResponse,
    AnalysisDetail,
    AnalysisListItem,
    ReportDraftPayload,
    ReportGenerateRequest,
    ReportGenerateResponse,
    ReportPreviewResponse,
)
from backend.app.schemas.audit import AuditEntryCreate, AuditEntryRead, CompleteReviewRequest
from backend.app.schemas.finding import FindingCreate, FindingRead, FindingUpdate
from backend.app.services.analysis_service import (
    complete_review,
    create_analysis_for_upload,
    get_analysis_detail,
    list_recent_analyses,
)
from backend.app.services.audit_service import append_audit_entry, list_audit_entries
from backend.app.services.domain_errors import DomainError, NotFoundError
from backend.app.services.finding_service import (
    create_manual_finding,
    delete_finding,
    finding_to_read,
    guest_finding_to_read,
    list_findings,
    update_finding,
)
from backend.app.services.guest_analysis_service import (
    complete_guest_review,
    create_guest_analysis_for_upload,
    create_manual_guest_finding,
    delete_guest_finding,
    guest_analysis_to_detail,
    update_guest_finding,
)
from backend.app.services.guest_workspace import GuestAuditEntry, GuestWorkspace
from backend.app.services.inference_service import process_analysis_inference, process_guest_inference
from backend.app.services.report_service import (
    build_report_preview,
    build_report_preview_from_guest,
    collect_pathology_crops,
    collect_pathology_crops_from_guest,
    collect_report_evidence_images,
    collect_report_evidence_images_from_guest,
    render_report_pdf,
    validate_report_generation_gate,
    _ensure_guest_report_allowed,
)

router = APIRouter(prefix="/analyses", tags=["analyses"])


def _parse_optional_int(raw: str | None) -> int | None:
    if raw is None or str(raw).strip() == "":
        return None
    return int(raw)


def _guest_audit_to_read(entry: GuestAuditEntry) -> AuditEntryRead:
    return AuditEntryRead(
        id=entry.id,
        reviewer=entry.reviewer,
        action=entry.action,
        action_type=entry.action_type,
        target_finding_id=entry.target_finding_id,
        before=entry.before,
        after=entry.after,
        timestamp=entry.timestamp,
    )


@router.post("", response_model=None)
async def upload_analysis(
    request: Request,
    session: DbSession,
    background_tasks: BackgroundTasks,
    model_registry: Annotated[object, Depends(get_model_registry)],
    workspace: Annotated[GuestWorkspace, Depends(get_guest_workspace)],
    principal: Annotated[AuthPrincipal | None, Depends(get_optional_principal)],
    patient_code: str = Form(...),
    patient_name: str = Form(...),
    age: str | None = Form(None),
    scan_date: str | None = Form(None),
    chief_complaint: str | None = Form(None),
    file: UploadFile = File(...),
) -> JSONResponse | AnalysisCreateResponse:
    """Upload OPG image + metadata; inference runs in the background."""
    age_val = _parse_optional_int(age)
    filename = file.filename or "upload.jpg"
    if principal is not None:
        analysis = await create_analysis_for_upload(
            session,
            patient_code=patient_code,
            patient_name=patient_name,
            age=age_val,
            scan_date_raw=scan_date,
            chief_complaint=chief_complaint,
            filename=filename,
            file_obj=file.file,
        )
        background_tasks.add_task(process_analysis_inference, analysis.id, model_registry)
        return AnalysisCreateResponse(id=analysis.id, status=analysis.status)

    session_id = get_guest_session_id(request) or uuid.uuid4()
    record = await create_guest_analysis_for_upload(
        workspace,
        session_id=session_id,
        patient_code=patient_code,
        patient_name=patient_name,
        age=age_val,
        scan_date_raw=scan_date,
        chief_complaint=chief_complaint,
        filename=filename,
        file_obj=file.file,
    )
    background_tasks.add_task(process_guest_inference, record.analysis_id, model_registry, workspace)
    settings = get_settings()
    body = AnalysisCreateResponse(id=record.analysis_id, status=record.status).model_dump(mode="json")
    resp = JSONResponse(content=body)
    resp.set_cookie(
        key=settings.guest_session_cookie_name,
        value=sign_guest_session_id(settings, session_id),
        max_age=settings.guest_session_ttl_seconds,
        httponly=True,
        samesite=settings.guest_cookie_samesite,
        secure=settings.guest_cookie_secure,
        path="/",
    )
    return resp


@router.get("", response_model=list[AnalysisListItem])
async def list_analyses(
    session: DbSession,
    workspace: Annotated[GuestWorkspace, Depends(get_guest_workspace)],
    principal: Annotated[AuthPrincipal | None, Depends(get_optional_principal)],
    request: Request,
    limit: int = Query(100, ge=1, le=500),
) -> list[AnalysisListItem]:
    """Recent analyses for the dashboard table."""
    if principal is not None:
        return await list_recent_analyses(session, limit=limit)
    sid = get_guest_session_id(request)
    if sid is None:
        return []
    return workspace.list_for_session(sid)


@router.get("/{analysis_id}", response_model=AnalysisDetail)
async def analysis_detail(
    session: DbSession,
    workspace: Annotated[GuestWorkspace, Depends(get_guest_workspace)],
    principal: Annotated[AuthPrincipal | None, Depends(get_optional_principal)],
    request: Request,
    analysis_id: uuid.UUID,
) -> AnalysisDetail:
    """Poll analysis status and aggregate metadata."""
    if principal is not None:
        return await get_analysis_detail(session, analysis_id)
    rec = workspace.get(analysis_id)
    if rec is None:
        raise NotFoundError("Analysis not found.")
    assert_guest_session_owns(request, rec.session_id)
    return guest_analysis_to_detail(rec)


@router.get("/{analysis_id}/image")
async def get_analysis_image(
    session: DbSession,
    workspace: Annotated[GuestWorkspace, Depends(get_guest_workspace)],
    principal: Annotated[AuthPrincipal | None, Depends(get_optional_principal)],
    request: Request,
    analysis_id: uuid.UUID,
    kind: ImageAssetKind = Query(default=ImageAssetKind.ORIGINAL),
) -> Response:
    """Stream a stored image artifact (BYTEA)."""
    if principal is not None:
        stmt = select(ImageAssetModel).where(
            ImageAssetModel.analysis_id == analysis_id,
            ImageAssetModel.kind == kind,
        )
        result = await session.execute(stmt)
        asset = result.scalar_one_or_none()
        if asset is None:
            raise NotFoundError("Image asset not found for this kind.")
        return Response(content=asset.data, media_type=asset.mime_type)
    rec = workspace.get(analysis_id)
    if rec is None:
        raise NotFoundError("Image asset not found for this kind.")
    assert_guest_session_owns(request, rec.session_id)
    asset = rec.assets.get(kind)
    if asset is None:
        raise NotFoundError("Image asset not found for this kind.")
    return Response(content=asset.data, media_type=asset.mime_type)


@router.get("/{analysis_id}/findings", response_model=list[FindingRead])
async def analysis_findings(
    session: DbSession,
    workspace: Annotated[GuestWorkspace, Depends(get_guest_workspace)],
    principal: Annotated[AuthPrincipal | None, Depends(get_optional_principal)],
    request: Request,
    analysis_id: uuid.UUID,
) -> list[FindingRead]:
    """List findings for viewer bootstrap."""
    if principal is not None:
        rows = await list_findings(session, analysis_id)
        return [finding_to_read(r) for r in rows]
    rec = workspace.get(analysis_id)
    if rec is None:
        raise NotFoundError("Analysis not found.")
    assert_guest_session_owns(request, rec.session_id)
    return [guest_finding_to_read(f) for f in rec.findings]


@router.post("/{analysis_id}/findings", response_model=FindingRead)
async def add_finding(
    session: DbSession,
    workspace: Annotated[GuestWorkspace, Depends(get_guest_workspace)],
    principal: Annotated[AuthPrincipal | None, Depends(get_optional_principal)],
    request: Request,
    analysis_id: uuid.UUID,
    payload: FindingCreate,
    x_reviewer: Annotated[str, Header(alias="X-Reviewer")],
) -> FindingRead:
    """Add a manual polygon finding."""
    reviewer = x_reviewer.strip()
    if not reviewer:
        raise DomainError("Header X-Reviewer is required.", status_code=422)
    if principal is not None:
        row = await create_manual_finding(session, analysis_id=analysis_id, payload=payload, reviewer=reviewer)
        await session.commit()
        await session.refresh(row)
        return finding_to_read(row)
    rec = workspace.get(analysis_id)
    if rec is None:
        raise NotFoundError("Analysis not found.")
    assert_guest_session_owns(request, rec.session_id)
    row = create_manual_guest_finding(workspace, rec, payload=payload, reviewer=reviewer)
    return guest_finding_to_read(row)


@router.get("/{analysis_id}/audit", response_model=list[AuditEntryRead])
async def list_audit(
    session: DbSession,
    workspace: Annotated[GuestWorkspace, Depends(get_guest_workspace)],
    principal: Annotated[AuthPrincipal | None, Depends(get_optional_principal)],
    request: Request,
    analysis_id: uuid.UUID,
) -> list[AuditEntryRead]:
    """Return audit log entries (newest first)."""
    if principal is not None:
        rows = await list_audit_entries(session, analysis_id)
        return [AuditEntryRead.model_validate(r) for r in rows]
    rec = workspace.get(analysis_id)
    if rec is None:
        raise NotFoundError("Analysis not found.")
    assert_guest_session_owns(request, rec.session_id)
    return [_guest_audit_to_read(e) for e in rec.audit]


@router.post("/{analysis_id}/audit", response_model=AuditEntryRead)
async def append_audit(
    session: DbSession,
    workspace: Annotated[GuestWorkspace, Depends(get_guest_workspace)],
    principal: Annotated[AuthPrincipal | None, Depends(get_optional_principal)],
    request: Request,
    analysis_id: uuid.UUID,
    payload: AuditEntryCreate,
) -> AuditEntryRead:
    """Append a client-side audit row (server still logs mutations separately)."""
    if principal is not None:
        row = await append_audit_entry(session, analysis_id=analysis_id, payload=payload)
        await session.commit()
        await session.refresh(row)
        return AuditEntryRead.model_validate(row)
    rec = workspace.get(analysis_id)
    if rec is None:
        raise NotFoundError("Analysis not found.")
    assert_guest_session_owns(request, rec.session_id)
    entry = workspace.append_audit(
        rec,
        reviewer=payload.reviewer,
        action=payload.action,
        action_type=payload.action_type,
        target_finding_id=payload.target_id,
        before=payload.before,
        after=payload.after,
    )
    return _guest_audit_to_read(entry)


@router.post("/{analysis_id}/complete-review", response_model=AnalysisDetail)
async def finish_review(
    session: DbSession,
    workspace: Annotated[GuestWorkspace, Depends(get_guest_workspace)],
    principal: Annotated[AuthPrincipal | None, Depends(get_optional_principal)],
    request: Request,
    analysis_id: uuid.UUID,
    body: CompleteReviewRequest,
) -> AnalysisDetail:
    """Finalize dentist review (HITL gate)."""
    if principal is not None:
        await complete_review(session, analysis_id=analysis_id, reviewer=body.reviewer)
        return await get_analysis_detail(session, analysis_id)
    rec = workspace.get(analysis_id)
    if rec is None:
        raise NotFoundError("Analysis not found.")
    assert_guest_session_owns(request, rec.session_id)
    complete_guest_review(workspace, analysis_id=analysis_id, reviewer=body.reviewer)
    return guest_analysis_to_detail(rec)


@router.get("/{analysis_id}/audit/export")
async def export_audit(
    session: DbSession,
    workspace: Annotated[GuestWorkspace, Depends(get_guest_workspace)],
    principal: Annotated[AuthPrincipal | None, Depends(get_optional_principal)],
    request: Request,
    analysis_id: uuid.UUID,
) -> Response:
    """Download audit log as JSON."""
    if principal is not None:
        rows = await list_audit_entries(session, analysis_id)
        payload = [AuditEntryRead.model_validate(r).model_dump(mode="json", by_alias=True) for r in rows]
        body = json.dumps(payload, indent=2).encode("utf-8")
        return Response(
            content=body,
            media_type="application/json",
            headers={"Content-Disposition": f'attachment; filename="audit-{analysis_id}.json"'},
        )
    rec = workspace.get(analysis_id)
    if rec is None:
        raise NotFoundError("Analysis not found.")
    assert_guest_session_owns(request, rec.session_id)
    payload = [_guest_audit_to_read(e).model_dump(mode="json", by_alias=True) for e in rec.audit]
    body = json.dumps(payload, indent=2).encode("utf-8")
    return Response(
        content=body,
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="audit-{analysis_id}.json"'},
    )


@router.post("/{analysis_id}/report/preview", response_model=ReportPreviewResponse)
async def preview_report(
    session: DbSession,
    workspace: Annotated[GuestWorkspace, Depends(get_guest_workspace)],
    principal: Annotated[AuthPrincipal | None, Depends(get_optional_principal)],
    request: Request,
    analysis_id: uuid.UUID,
    payload: ReportDraftPayload,
) -> ReportPreviewResponse:
    """Build normalized preview content for report editor."""
    if principal is not None:
        preview = await build_report_preview(session, analysis_id=analysis_id, payload=payload)
        await append_audit_entry(
            session,
            analysis_id=analysis_id,
            payload=AuditEntryCreate(
                reviewer=payload.reviewer_confirmation or "system",
                action="Previewed PDF report draft",
                action_type=AuditActionType.EXPORT,
            ),
        )
        await session.commit()
        return preview
    rec = workspace.get(analysis_id)
    if rec is None:
        raise NotFoundError("Analysis not found.")
    assert_guest_session_owns(request, rec.session_id)
    preview = build_report_preview_from_guest(rec, payload)
    workspace.append_audit(
        rec,
        reviewer=payload.reviewer_confirmation or "system",
        action="Previewed PDF report draft",
        action_type=AuditActionType.EXPORT,
    )
    return preview


@router.post("/{analysis_id}/report/generate", response_model=ReportGenerateResponse)
async def generate_report(
    session: DbSession,
    workspace: Annotated[GuestWorkspace, Depends(get_guest_workspace)],
    principal: Annotated[AuthPrincipal | None, Depends(get_optional_principal)],
    request: Request,
    analysis_id: uuid.UUID,
    payload: ReportGenerateRequest,
) -> ReportGenerateResponse:
    """Generate and persist a PDF report for a reviewed analysis."""
    generated_at = datetime.now(timezone.utc)
    if principal is not None:
        analysis = await validate_report_generation_gate(session, analysis_id=analysis_id)
        preview = await build_report_preview(session, analysis_id=analysis_id, payload=payload)
        if preview.accepted_findings_count == 0:
            raise DomainError("Cannot generate report without accepted findings.", status_code=422)

        evidence_images = await collect_report_evidence_images(
            session,
            analysis_id=analysis_id,
            include_images=payload.include_images,
        )
        pathology_crops = await collect_pathology_crops(
            session,
            analysis_id=analysis_id,
            include_images=payload.include_images,
        )
        pdf_bytes = render_report_pdf(
            preview=preview,
            reviewer=payload.reviewer.strip(),
            evidence_images=evidence_images,
            pathology_crops=pathology_crops,
            generated_at=generated_at,
        )
        filename = f"report-{analysis_id}.pdf"
        from sqlalchemy import delete

        await session.execute(delete(ReportAsset).where(ReportAsset.analysis_id == analysis_id))
        report = ReportAsset(
            analysis_id=analysis_id,
            reviewer=payload.reviewer.strip(),
            filename=filename,
            mime_type="application/pdf",
            data=pdf_bytes,
        )
        session.add(report)
        analysis.status = AnalysisStatus.REPORT_GENERATED
        analysis.reviewer = payload.reviewer.strip()
        await append_audit_entry(
            session,
            analysis_id=analysis_id,
            payload=AuditEntryCreate(
                reviewer=payload.reviewer.strip(),
                action=f"Generated PDF report {filename}",
                action_type=AuditActionType.EXPORT,
            ),
        )
        await session.commit()
        await session.refresh(report)
        return ReportGenerateResponse(
            report_id=report.id,
            analysis_id=analysis_id,
            status=analysis.status,
            generated_at=report.created_at or generated_at,
            filename=report.filename,
            download_url=f"/api/v1/analyses/{analysis_id}/report/download",
        )

    rec = workspace.get(analysis_id)
    if rec is None:
        raise NotFoundError("Analysis not found.")
    assert_guest_session_owns(request, rec.session_id)
    _ensure_guest_report_allowed(rec)
    preview = build_report_preview_from_guest(rec, payload)
    if preview.accepted_findings_count == 0:
        raise DomainError("Cannot generate report without accepted findings.", status_code=422)
    evidence_images = collect_report_evidence_images_from_guest(
        rec,
        include_images=payload.include_images,
    )
    pathology_crops = collect_pathology_crops_from_guest(
        rec,
        include_images=payload.include_images,
    )
    pdf_bytes = render_report_pdf(
        preview=preview,
        reviewer=payload.reviewer.strip(),
        evidence_images=evidence_images,
        pathology_crops=pathology_crops,
        generated_at=generated_at,
    )
    filename = f"report-{analysis_id}.pdf"
    rec.report_pdf = pdf_bytes
    rec.status = AnalysisStatus.REPORT_GENERATED
    rec.reviewer = payload.reviewer.strip()
    rid = uuid.uuid4()
    workspace.append_audit(
        rec,
        reviewer=payload.reviewer.strip(),
        action=f"Generated PDF report {filename}",
        action_type=AuditActionType.EXPORT,
    )
    return ReportGenerateResponse(
        report_id=rid,
        analysis_id=analysis_id,
        status=rec.status,
        generated_at=generated_at,
        filename=filename,
        download_url=f"/api/v1/analyses/{analysis_id}/report/download",
    )


@router.get("/{analysis_id}/report/download")
async def download_report(
    session: DbSession,
    workspace: Annotated[GuestWorkspace, Depends(get_guest_workspace)],
    principal: Annotated[AuthPrincipal | None, Depends(get_optional_principal)],
    request: Request,
    analysis_id: uuid.UUID,
) -> Response:
    """Download the latest generated PDF report for an analysis."""
    if principal is not None:
        stmt = (
            select(ReportAsset)
            .where(ReportAsset.analysis_id == analysis_id)
            .order_by(desc(ReportAsset.created_at))
            .limit(1)
        )
        report = (await session.execute(stmt)).scalar_one_or_none()
        if report is None:
            raise NotFoundError("Report not generated for this analysis.")
        await append_audit_entry(
            session,
            analysis_id=analysis_id,
            payload=AuditEntryCreate(
                reviewer="system",
                action=f"Downloaded PDF report {report.filename}",
                action_type=AuditActionType.EXPORT,
            ),
        )
        await session.commit()
        return Response(
            content=report.data,
            media_type=report.mime_type,
            headers={"Content-Disposition": f'attachment; filename="{report.filename}"'},
        )
    rec = workspace.get(analysis_id)
    if rec is None:
        raise NotFoundError("Report not generated for this analysis.")
    assert_guest_session_owns(request, rec.session_id)
    if not rec.report_pdf:
        raise NotFoundError("Report not generated for this analysis.")
    filename = f"report-{analysis_id}.pdf"
    workspace.append_audit(
        rec,
        reviewer="system",
        action=f"Downloaded PDF report {filename}",
        action_type=AuditActionType.EXPORT,
    )
    return Response(
        content=rec.report_pdf,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
