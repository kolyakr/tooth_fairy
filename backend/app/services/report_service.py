"""PDF report generation and preview helpers."""

from __future__ import annotations

import io
import uuid
from datetime import datetime, timezone
from typing import Any

from PIL import Image
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from backend.app.db.models import Analysis, AnalysisStatus, Finding, FindingSource, ImageAssetKind, Patient
from backend.app.schemas.analysis import (
    ReportDraftPayload,
    ReportFindingPreview,
    ReportPreviewResponse,
)
from backend.app.services.domain_errors import NotFoundError, ValidationDomainError
from backend.app.services.guest_workspace import GuestAnalysisRecord
from backend.app.services.report_pdf_templates import render_template_b_letterhead

_EVIDENCE_IMAGE_PRIORITY: list[ImageAssetKind] = [
    ImageAssetKind.TEETH_CLASSIFICATION_OVERLAY,
    ImageAssetKind.PERIAPICAL_QUADRANTS_OVERLAY,
    ImageAssetKind.TEETH_OVERLAY,
    ImageAssetKind.QUADRANTS_OVERLAY,
    ImageAssetKind.ORIGINAL,
]

_PATHOLOGY_SECTIONS: dict[str, str] = {
    "Periapical Lesion": "Periapical Findings",
    "Impacted": "Impacted Findings",
    "Caries": "Caries Findings",
}


def _normalized_text(value: str) -> str:
    """Normalize user-provided text for deterministic preview/PDF output."""
    return " ".join(value.strip().split())


def _build_sections(payload: ReportDraftPayload) -> list[dict[str, str]]:
    """Build canonical report sections from draft payload."""
    sections: list[dict[str, str]] = [
        {"title": "Clinical Summary", "body": _normalized_text(payload.clinical_summary)},
        {"title": "Impression", "body": _normalized_text(payload.impression)},
        {"title": "Recommendations", "body": _normalized_text(payload.recommendations)},
    ]
    if payload.reviewer_confirmation and payload.reviewer_confirmation.strip():
        sections.append(
            {
                "title": "Reviewer Confirmation",
                "body": _normalized_text(payload.reviewer_confirmation),
            }
        )
    return sections


def _finding_bounds(row: Any) -> tuple[int, int, int, int] | None:
    """Resolve finding crop bounds from box or polygon geometry."""
    if row.box_xyxy and isinstance(row.box_xyxy, list) and len(row.box_xyxy) == 4:
        x1, y1, x2, y2 = row.box_xyxy
        return int(float(x1)), int(float(y1)), int(float(x2)), int(float(y2))
    if row.polygon and isinstance(row.polygon, list):
        xs: list[float] = []
        ys: list[float] = []
        for p in row.polygon:
            if isinstance(p, dict) and "x" in p and "y" in p:
                xs.append(float(p["x"]))
                ys.append(float(p["y"]))
        if xs and ys:
            return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))
    return None


def _padded_crop_bytes(
    image_bytes: bytes,
    *,
    bounds: tuple[int, int, int, int],
    padding_px: int = 24,
    min_size: int = 56,
) -> bytes | None:
    """Create padded PNG crop bytes from source image and geometry bounds."""
    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            rgb = img.convert("RGB")
            w, h = rgb.size
            x1, y1, x2, y2 = bounds
            x1 = max(0, x1 - padding_px)
            y1 = max(0, y1 - padding_px)
            x2 = min(w, x2 + padding_px)
            y2 = min(h, y2 + padding_px)
            if x2 <= x1 or y2 <= y1:
                return None
            if (x2 - x1) < min_size:
                extra = (min_size - (x2 - x1)) // 2
                x1 = max(0, x1 - extra)
                x2 = min(w, x2 + extra)
            if (y2 - y1) < min_size:
                extra = (min_size - (y2 - y1)) // 2
                y1 = max(0, y1 - extra)
                y2 = min(h, y2 + extra)
            crop = rgb.crop((x1, y1, x2, y2))
            out = io.BytesIO()
            crop.save(out, format="PNG", optimize=True)
            return out.getvalue()
    except Exception:
        return None


async def build_report_preview(
    session: AsyncSession,
    *,
    analysis_id: uuid.UUID,
    payload: ReportDraftPayload,
) -> ReportPreviewResponse:
    """Return normalized preview data used both by UI and PDF renderer."""
    stmt = (
        select(Analysis)
        .where(Analysis.id == analysis_id)
        .options(
            selectinload(Analysis.patient),
            selectinload(Analysis.findings),
            selectinload(Analysis.image_assets),
        )
    )
    analysis = (await session.execute(stmt)).scalar_one_or_none()
    if analysis is None:
        raise NotFoundError("Analysis not found.")

    patient = analysis.patient
    if patient is None:
        raise ValidationDomainError("Patient context is missing for this analysis.")

    accepted_rows = [
        f
        for f in analysis.findings
        if f.accepted and f.source in {FindingSource.AI, FindingSource.MANUAL}
    ]
    accepted_rows.sort(key=lambda row: (row.tooth_label, row.finding_class, -row.confidence))
    accepted = [
        ReportFindingPreview(
            id=row.id,
            tooth_label=row.tooth_label,
            finding=row.finding_class,
            confidence=float(row.confidence),
            layer=row.layer.value,
        )
        for row in accepted_rows
    ]
    image_kinds = sorted(
        {
            asset.kind.value
            for asset in analysis.image_assets
            if payload.include_images and asset.kind != ImageAssetKind.ORIGINAL
        }
    )
    return ReportPreviewResponse(
        analysis_id=analysis.id,
        patient_name=patient.name,
        patient_code=patient.patient_code,
        scan_date=analysis.scan_date,
        accepted_findings_count=len(accepted),
        accepted_findings=accepted,
        sections=_build_sections(payload),
        image_kinds=image_kinds,
    )


def _ensure_report_allowed(analysis: Analysis) -> None:
    """Validate report generation gate based on analysis status."""
    if analysis.status not in {AnalysisStatus.REVIEWED, AnalysisStatus.REPORT_GENERATED}:
        raise ValidationDomainError("Report generation requires analysis status 'Reviewed'.")


async def validate_report_generation_gate(
    session: AsyncSession,
    *,
    analysis_id: uuid.UUID,
) -> Analysis:
    """Ensure analysis exists and is allowed to generate reports."""
    stmt = select(Analysis).where(Analysis.id == analysis_id)
    analysis = (await session.execute(stmt)).scalar_one_or_none()
    if analysis is None:
        raise NotFoundError("Analysis not found.")
    _ensure_report_allowed(analysis)
    return analysis


async def collect_report_evidence_images(
    session: AsyncSession,
    *,
    analysis_id: uuid.UUID,
    include_images: bool,
    max_images: int = 3,
) -> list[tuple[str, bytes]]:
    """Collect prioritized overlay/original assets for embedding into the PDF body."""
    if not include_images:
        return []

    stmt = (
        select(Analysis)
        .where(Analysis.id == analysis_id)
        .options(selectinload(Analysis.image_assets))
    )
    analysis = (await session.execute(stmt)).scalar_one_or_none()
    if analysis is None:
        return []

    by_kind = {asset.kind: asset for asset in analysis.image_assets}
    selected: list[tuple[str, bytes]] = []
    for kind in _EVIDENCE_IMAGE_PRIORITY:
        asset = by_kind.get(kind)
        if asset is None:
            continue
        selected.append((kind.value, asset.data))
        if len(selected) >= max_images:
            break
    return selected


async def collect_pathology_crops(
    session: AsyncSession,
    *,
    analysis_id: uuid.UUID,
    include_images: bool,
) -> dict[str, list[dict[str, Any]]]:
    """Collect all pathology finding crops grouped by section title."""
    grouped: dict[str, list[dict[str, Any]]] = {title: [] for title in _PATHOLOGY_SECTIONS.values()}
    if not include_images:
        return grouped

    stmt = (
        select(Analysis)
        .where(Analysis.id == analysis_id)
        .options(selectinload(Analysis.image_assets), selectinload(Analysis.findings))
    )
    analysis = (await session.execute(stmt)).scalar_one_or_none()
    if analysis is None:
        return grouped

    asset_by_kind = {a.kind: a for a in analysis.image_assets}
    source_asset = asset_by_kind.get(ImageAssetKind.ORIGINAL) or next(iter(analysis.image_assets), None)
    if source_asset is None:
        return grouped

    for row in analysis.findings:
        if not row.accepted:
            continue
        section = _PATHOLOGY_SECTIONS.get(row.finding_class)
        if section is None:
            continue
        bounds = _finding_bounds(row)
        crop = _padded_crop_bytes(source_asset.data, bounds=bounds) if bounds else None
        grouped[section].append(
            {
                "tooth": row.tooth_label,
                "finding": row.finding_class,
                "confidence": float(row.confidence),
                "image": crop,
            }
        )
    return grouped


def _ensure_guest_report_allowed(record: GuestAnalysisRecord) -> None:
    if record.status not in {AnalysisStatus.REVIEWED, AnalysisStatus.REPORT_GENERATED}:
        raise ValidationDomainError("Report generation requires analysis status 'Reviewed'.")


def build_report_preview_from_guest(
    record: GuestAnalysisRecord,
    payload: ReportDraftPayload,
) -> ReportPreviewResponse:
    """Build ``ReportPreviewResponse`` from in-memory guest analysis."""
    accepted_rows = [
        f
        for f in record.findings
        if f.accepted and f.source in {FindingSource.AI, FindingSource.MANUAL}
    ]
    accepted_rows.sort(key=lambda row: (row.tooth_label, row.finding_class, -row.confidence))
    accepted = [
        ReportFindingPreview(
            id=row.id,
            tooth_label=row.tooth_label,
            finding=row.finding_class,
            confidence=float(row.confidence),
            layer=row.layer.value,
        )
        for row in accepted_rows
    ]
    image_kinds = sorted(
        {
            k.value
            for k in record.assets
            if payload.include_images and k != ImageAssetKind.ORIGINAL
        }
    )
    return ReportPreviewResponse(
        analysis_id=record.analysis_id,
        patient_name=record.patient_name,
        patient_code=record.patient_code,
        scan_date=record.scan_date,
        accepted_findings_count=len(accepted),
        accepted_findings=accepted,
        sections=_build_sections(payload),
        image_kinds=image_kinds,
    )


def collect_report_evidence_images_from_guest(
    record: GuestAnalysisRecord,
    *,
    include_images: bool,
    max_images: int = 3,
) -> list[tuple[str, bytes]]:
    if not include_images:
        return []
    by_kind = record.assets
    selected: list[tuple[str, bytes]] = []
    for kind in _EVIDENCE_IMAGE_PRIORITY:
        asset = by_kind.get(kind)
        if asset is None:
            continue
        selected.append((kind.value, asset.data))
        if len(selected) >= max_images:
            break
    return selected


def collect_pathology_crops_from_guest(
    record: GuestAnalysisRecord,
    *,
    include_images: bool,
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {title: [] for title in _PATHOLOGY_SECTIONS.values()}
    if not include_images:
        return grouped
    source_asset = record.assets.get(ImageAssetKind.ORIGINAL) or next(iter(record.assets.values()), None)
    if source_asset is None:
        return grouped
    for row in record.findings:
        if not row.accepted:
            continue
        section = _PATHOLOGY_SECTIONS.get(row.finding_class)
        if section is None:
            continue
        bounds = _finding_bounds(row)
        crop = _padded_crop_bytes(source_asset.data, bounds=bounds) if bounds else None
        grouped[section].append(
            {
                "tooth": row.tooth_label,
                "finding": row.finding_class,
                "confidence": float(row.confidence),
                "image": crop,
            }
        )
    return grouped


def render_report_pdf(
    *,
    preview: ReportPreviewResponse,
    reviewer: str,
    evidence_images: list[tuple[str, bytes]] | None = None,
    pathology_crops: dict[str, list[dict[str, Any]]] | None = None,
    generated_at: datetime | None = None,
) -> bytes:
    """Render a deterministic PDF binary (practice letterhead layout)."""
    return render_template_b_letterhead(
        preview=preview,
        reviewer=reviewer,
        evidence_images=evidence_images,
        pathology_crops=pathology_crops,
        generated_at=generated_at,
    )
