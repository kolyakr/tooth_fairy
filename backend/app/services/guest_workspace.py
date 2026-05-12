"""In-memory analysis state for unauthenticated (guest) sessions."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from backend.app.core.config import Settings
from backend.app.db.models import (
    AlertLevel,
    AnalysisStatus,
    AuditActionType,
    FindingLayer,
    FindingSource,
    ImageAssetKind,
)
from backend.app.schemas.analysis import AnalysisListItem
from backend.app.services.domain_errors import NotFoundError, ValidationDomainError


@dataclass
class GuestImageAsset:
    """Raster payload for one ``ImageAssetKind``."""

    kind: ImageAssetKind
    mime_type: str
    data: bytes
    width: int
    height: int


@dataclass
class GuestFinding:
    """Finding row compatible with ``FindingRead`` mapping."""

    id: uuid.UUID
    analysis_id: uuid.UUID
    tooth_label: str
    finding_class: str
    layer: FindingLayer
    confidence: float
    box_xyxy: list[float] | None
    polygon: list[dict[str, float]]
    accepted: bool
    source: FindingSource
    created_at: datetime
    updated_at: datetime


@dataclass
class GuestAuditEntry:
    """In-memory audit row."""

    id: uuid.UUID
    reviewer: str
    action: str
    action_type: AuditActionType
    target_finding_id: uuid.UUID | None
    before: dict[str, Any] | None
    after: dict[str, Any] | None
    timestamp: datetime


@dataclass
class GuestAnalysisRecord:
    """Full guest analysis graph (no PostgreSQL)."""

    analysis_id: uuid.UUID
    session_id: uuid.UUID
    patient_code: str
    patient_name: str
    patient_age: int | None
    filename: str
    scan_date: datetime | None
    chief_complaint: str | None
    status: AnalysisStatus
    alert_level: AlertLevel | None
    error: str | None
    reviewer: str | None
    created_at: datetime
    completed_at: datetime | None
    assets: dict[ImageAssetKind, GuestImageAsset] = field(default_factory=dict)
    findings: list[GuestFinding] = field(default_factory=list)
    audit: list[GuestAuditEntry] = field(default_factory=list)
    report_pdf: bytes | None = None


class GuestWorkspace:
    """Process-global guest store with TTL eviction."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._analyses: dict[uuid.UUID, GuestAnalysisRecord] = {}
        self._session_index: dict[uuid.UUID, list[uuid.UUID]] = {}
        self._finding_index: dict[uuid.UUID, uuid.UUID] = {}

    def _now(self) -> datetime:
        return datetime.now(timezone.utc)

    def _evict_stale(self) -> None:
        ttl = self._settings.guest_session_ttl_seconds
        cutoff = self._now().timestamp() - ttl
        stale_ids: list[uuid.UUID] = []
        for aid, rec in self._analyses.items():
            if rec.created_at.timestamp() < cutoff:
                stale_ids.append(aid)
        for aid in stale_ids:
            self._remove_analysis_unlocked(aid)

    def _remove_analysis_unlocked(self, analysis_id: uuid.UUID) -> None:
        rec = self._analyses.pop(analysis_id, None)
        if rec is None:
            return
        for f in rec.findings:
            self._finding_index.pop(f.id, None)
        lst = self._session_index.get(rec.session_id)
        if lst:
            self._session_index[rec.session_id] = [x for x in lst if x != analysis_id]
            if not self._session_index[rec.session_id]:
                del self._session_index[rec.session_id]

    def register_analysis(
        self,
        *,
        session_id: uuid.UUID,
        record: GuestAnalysisRecord,
    ) -> None:
        """Insert a new guest analysis and index it."""
        self._evict_stale()
        if len(self._session_index.get(session_id, [])) >= self._settings.guest_max_analyses_per_session:
            raise ValidationDomainError("Guest session analysis limit reached.")
        self._analyses[record.analysis_id] = record
        self._session_index.setdefault(session_id, []).append(record.analysis_id)
        for f in record.findings:
            self._finding_index[f.id] = record.analysis_id

    def get(self, analysis_id: uuid.UUID) -> GuestAnalysisRecord | None:
        self._evict_stale()
        return self._analyses.get(analysis_id)

    def require(self, analysis_id: uuid.UUID) -> GuestAnalysisRecord:
        rec = self.get(analysis_id)
        if rec is None:
            raise NotFoundError("Analysis not found.")
        return rec

    def list_for_session(self, session_id: uuid.UUID) -> list[AnalysisListItem]:
        self._evict_stale()
        ids = list(self._session_index.get(session_id, []))
        out: list[AnalysisListItem] = []
        for aid in reversed(ids):
            rec = self._analyses.get(aid)
            if rec is None:
                continue
            out.append(
                AnalysisListItem(
                    id=rec.analysis_id,
                    patient_name=rec.patient_name,
                    patient_id=rec.patient_code,
                    scan_date=rec.scan_date,
                    status=rec.status,
                    alert_level=rec.alert_level,
                )
            )
        return out

    def get_analysis_for_finding(self, finding_id: uuid.UUID) -> tuple[GuestAnalysisRecord, GuestFinding] | None:
        self._evict_stale()
        aid = self._finding_index.get(finding_id)
        if aid is None:
            return None
        rec = self._analyses.get(aid)
        if rec is None:
            return None
        for f in rec.findings:
            if f.id == finding_id:
                return rec, f
        return None

    def append_finding(self, record: GuestAnalysisRecord, finding: GuestFinding) -> None:
        record.findings.append(finding)
        self._finding_index[finding.id] = record.analysis_id

    def remove_finding(self, record: GuestAnalysisRecord, finding_id: uuid.UUID) -> None:
        record.findings = [f for f in record.findings if f.id != finding_id]
        self._finding_index.pop(finding_id, None)

    def extend_findings(self, record: GuestAnalysisRecord, new_findings: list[GuestFinding]) -> None:
        for f in new_findings:
            record.findings.append(f)
            self._finding_index[f.id] = record.analysis_id

    def append_audit(
        self,
        record: GuestAnalysisRecord,
        *,
        reviewer: str,
        action: str,
        action_type: AuditActionType,
        target_finding_id: uuid.UUID | None = None,
        before: dict[str, Any] | None = None,
        after: dict[str, Any] | None = None,
    ) -> GuestAuditEntry:
        entry = GuestAuditEntry(
            id=uuid.uuid4(),
            reviewer=reviewer,
            action=action,
            action_type=action_type,
            target_finding_id=target_finding_id,
            before=before,
            after=after,
            timestamp=self._now(),
        )
        record.audit.insert(0, entry)
        return entry
