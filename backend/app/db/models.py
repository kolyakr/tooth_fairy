"""Persistence models for patients, analyses, findings, and audit trails."""

from __future__ import annotations

import enum
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any, List, Optional

from sqlalchemy import (
    JSON,
    DateTime,
    Enum as SAEnum,
    ForeignKey,
    Integer,
    LargeBinary,
    String,
    Text,
    Uuid,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.db.base import Base

if TYPE_CHECKING:
    pass


class AnalysisStatus(str, enum.Enum):
    """Lifecycle state of an AI-assisted scan."""

    PENDING_AI = "Pending AI"
    REVIEWING = "Reviewing"
    REVIEWED = "Reviewed"
    REPORT_GENERATED = "Report Generated"
    FAILED = "Failed"


class AlertLevel(str, enum.Enum):
    """Aggregate severity hint for dashboard highlighting."""

    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"


class ImageAssetKind(str, enum.Enum):
    """Kinds of stored raster artifacts."""

    ORIGINAL = "original"
    QUADRANTS_OVERLAY = "quadrants_overlay"
    QUADRANTS_GRID = "quadrants_grid"
    TEETH_OVERLAY = "teeth_overlay"
    PERIAPICAL_FULL_OVERLAY = "periapical_full_overlay"
    PERIAPICAL_QUADRANTS_OVERLAY = "periapical_quadrants_overlay"
    TEETH_CLASSIFICATION_OVERLAY = "teeth_classification_overlay"


class FindingLayer(str, enum.Enum):
    """Viewer layer grouping."""

    QUADRANTS = "quadrants"
    TEETH = "teeth"
    PERIAPICAL = "periapical"


class FindingSource(str, enum.Enum):
    """Origin of a detection record."""

    AI = "ai"
    MANUAL = "manual"


class AuditActionType(str, enum.Enum):
    """Coarse audit classification."""

    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    REVIEW = "review"
    EXPORT = "export"
    SYSTEM = "system"


class Patient(Base):
    """Demographics captured for a scan subject."""

    __tablename__ = "patients"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    patient_code: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(255))
    age: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    analyses: Mapped[List["Analysis"]] = relationship(back_populates="patient", cascade="all, delete-orphan")


class Analysis(Base):
    """One panoramic analysis session."""

    __tablename__ = "analyses"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    patient_id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), ForeignKey("patients.id", ondelete="CASCADE"))
    filename: Mapped[str] = mapped_column(String(512))
    scan_date: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    chief_complaint: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    status: Mapped[AnalysisStatus] = mapped_column(
        SAEnum(AnalysisStatus, values_callable=lambda x: [e.value for e in x], name="analysis_status"),
        default=AnalysisStatus.PENDING_AI,
    )
    alert_level: Mapped[Optional[AlertLevel]] = mapped_column(
        SAEnum(AlertLevel, values_callable=lambda x: [e.value for e in x], name="alert_level"),
        nullable=True,
    )
    reviewer: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    patient: Mapped["Patient"] = relationship(back_populates="analyses")
    image_assets: Mapped[List["ImageAsset"]] = relationship(back_populates="analysis", cascade="all, delete-orphan")
    findings: Mapped[List["Finding"]] = relationship(back_populates="analysis", cascade="all, delete-orphan")
    audit_entries: Mapped[List["AuditEntry"]] = relationship(back_populates="analysis", cascade="all, delete-orphan")
    report_assets: Mapped[List["ReportAsset"]] = relationship(
        back_populates="analysis", cascade="all, delete-orphan"
    )


class ImageAsset(Base):
    """Binary image payload stored in-database (BYTEA)."""

    __tablename__ = "image_assets"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    analysis_id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), ForeignKey("analyses.id", ondelete="CASCADE"))
    kind: Mapped[ImageAssetKind] = mapped_column(
        SAEnum(ImageAssetKind, values_callable=lambda x: [e.value for e in x], name="image_asset_kind"),
    )
    mime_type: Mapped[str] = mapped_column(String(128))
    data: Mapped[bytes] = mapped_column(LargeBinary)
    width: Mapped[int] = mapped_column(Integer)
    height: Mapped[int] = mapped_column(Integer)

    analysis: Mapped["Analysis"] = relationship(back_populates="image_assets")


class Finding(Base):
    """A single tooth-linked annotation (AI or manual)."""

    __tablename__ = "findings"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    analysis_id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), ForeignKey("analyses.id", ondelete="CASCADE"))
    tooth_label: Mapped[str] = mapped_column(String(64))
    finding_class: Mapped[str] = mapped_column(String(255))
    layer: Mapped[FindingLayer] = mapped_column(
        SAEnum(FindingLayer, values_callable=lambda x: [e.value for e in x], name="finding_layer"),
    )
    confidence: Mapped[float] = mapped_column()
    box_xyxy: Mapped[Optional[Any]] = mapped_column(JSON, nullable=True)
    polygon: Mapped[Optional[Any]] = mapped_column(JSON, nullable=True)
    accepted: Mapped[bool] = mapped_column(default=True)
    source: Mapped[FindingSource] = mapped_column(
        SAEnum(FindingSource, values_callable=lambda x: [e.value for e in x], name="finding_source"),
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    analysis: Mapped["Analysis"] = relationship(back_populates="findings")


class AuditEntry(Base):
    """Immutable audit log row for HITL traceability."""

    __tablename__ = "audit_entries"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    analysis_id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), ForeignKey("analyses.id", ondelete="CASCADE"))
    reviewer: Mapped[str] = mapped_column(String(255))
    action: Mapped[str] = mapped_column(Text)
    action_type: Mapped[AuditActionType] = mapped_column(
        SAEnum(AuditActionType, values_callable=lambda x: [e.value for e in x], name="audit_action_type"),
    )
    target_finding_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("findings.id", ondelete="SET NULL"), nullable=True
    )
    before: Mapped[Optional[Any]] = mapped_column(JSON, nullable=True)
    after: Mapped[Optional[Any]] = mapped_column(JSON, nullable=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    analysis: Mapped["Analysis"] = relationship(back_populates="audit_entries")


class ReportAsset(Base):
    """Generated PDF report payload stored in-database (BYTEA)."""

    __tablename__ = "report_assets"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    analysis_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("analyses.id", ondelete="CASCADE"), index=True
    )
    reviewer: Mapped[str] = mapped_column(String(255))
    filename: Mapped[str] = mapped_column(String(512))
    mime_type: Mapped[str] = mapped_column(String(128), default="application/pdf")
    data: Mapped[bytes] = mapped_column(LargeBinary)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    analysis: Mapped["Analysis"] = relationship(back_populates="report_assets")
