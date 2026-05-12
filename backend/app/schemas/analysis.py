"""Analysis API schemas (dashboard + polling)."""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from backend.app.db.models import AlertLevel, AnalysisStatus


class AnalysisCreateResponse(BaseModel):
    """Returned immediately after upload while inference runs."""

    id: uuid.UUID
    status: AnalysisStatus


class AnalysisListItem(BaseModel):
    """Row shape compatible with the dashboard table."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    patient_name: str
    patient_id: str = Field(..., description="External patient code (e.g. P-4021).")
    scan_date: datetime | None
    status: AnalysisStatus
    alert_level: AlertLevel | None


class AnalysisDetail(BaseModel):
    """Full analysis record for polling and viewer bootstrap."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    patient_id: uuid.UUID
    filename: str
    scan_date: datetime | None
    chief_complaint: str | None
    status: AnalysisStatus
    alert_level: AlertLevel | None
    reviewer: str | None
    error: str | None
    created_at: datetime
    completed_at: datetime | None
    findings_count: int = 0
    image_kinds: list[str] = Field(default_factory=list)


class ReportDraftPayload(BaseModel):
    """Dentist-authored narrative input for report preview/generation."""

    clinical_summary: str = Field(..., min_length=1)
    impression: str = Field(..., min_length=1)
    recommendations: str = Field(..., min_length=1)
    reviewer_confirmation: str | None = None
    include_images: bool = True


class ReportGenerateRequest(ReportDraftPayload):
    """Request body to render and persist a PDF report."""

    reviewer: str = Field(..., min_length=1)


class ReportFindingPreview(BaseModel):
    """Accepted finding row used in report preview."""

    id: uuid.UUID
    tooth_label: str
    finding: str
    confidence: float
    layer: str


class ReportPreviewResponse(BaseModel):
    """Preview payload used by frontend editor before PDF generation."""

    analysis_id: uuid.UUID
    patient_name: str
    patient_code: str
    scan_date: datetime | None
    accepted_findings_count: int
    accepted_findings: list[ReportFindingPreview] = Field(default_factory=list)
    sections: list[dict[str, str]] = Field(default_factory=list)
    image_kinds: list[str] = Field(default_factory=list)


class ReportGenerateResponse(BaseModel):
    """Metadata returned after generating and storing a PDF report."""

    report_id: uuid.UUID
    analysis_id: uuid.UUID
    status: AnalysisStatus
    generated_at: datetime
    filename: str
    download_url: str
