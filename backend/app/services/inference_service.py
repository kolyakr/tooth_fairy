"""Background inference: wraps ``run_pipeline`` and persists artifacts."""

from __future__ import annotations

import asyncio
import tempfile
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.config import get_settings
from backend.app.core.database import AsyncSessionLocal
from backend.app.db.models import (
    AlertLevel,
    Analysis,
    AnalysisStatus,
    AuditActionType,
    Finding,
    FindingLayer,
    FindingSource,
    ImageAsset,
    ImageAssetKind,
)
from backend.app.image_utils import mime_and_dimensions
from backend.app.ml.modeling_imports import setup_modeling_path
from backend.app.services.audit_service import append_system_audit
from backend.app.services.guest_workspace import GuestFinding, GuestImageAsset, GuestWorkspace

_PIPELINE_FILE_TO_KIND: dict[str, ImageAssetKind] = {
    "quadrants_overlay": ImageAssetKind.QUADRANTS_OVERLAY,
    "quadrants_grid": ImageAssetKind.QUADRANTS_GRID,
    "teeth_overlay": ImageAssetKind.TEETH_OVERLAY,
    "periapical_quadrants_overlay": ImageAssetKind.PERIAPICAL_QUADRANTS_OVERLAY,
    "teeth_classification_overlay": ImageAssetKind.TEETH_CLASSIFICATION_OVERLAY,
    "original": ImageAssetKind.ORIGINAL,
}


def _box_to_polygon(box: list[float]) -> list[dict[str, float]]:
    """Turn ``xyxy`` into a closed quadrilateral for the viewer."""
    x1, y1, x2, y2 = box
    return [
        {"x": float(x1), "y": float(y1)},
        {"x": float(x2), "y": float(y1)},
        {"x": float(x2), "y": float(y2)},
        {"x": float(x1), "y": float(y2)},
    ]


def _mask_to_polygon(mask_xy: list[list[float]] | None) -> list[dict[str, float]]:
    """Normalize Ultralytics mask polygon to viewer vertices."""
    if not mask_xy:
        return []
    return [{"x": float(x), "y": float(y)} for x, y in mask_xy]


def _compute_alert_level(predictions: dict[str, list[dict]]) -> AlertLevel:
    """Simple heuristic for dashboard severity (tunable)."""
    scores: list[float] = []
    peri_count = 0
    for key, rows in predictions.items():
        for row in rows:
            c = row.get("confidence")
            if isinstance(c, (int, float)):
                scores.append(float(c))
        if key == "periapical":
            peri_count += len(rows)
    max_conf = max(scores) if scores else 0.0
    if max_conf >= 0.85 or peri_count >= 2:
        return AlertLevel.HIGH
    if max_conf >= 0.55 or len(scores) > 25:
        return AlertLevel.MEDIUM
    return AlertLevel.LOW


def _predictions_to_findings(analysis_id: uuid.UUID, predictions: dict[str, list[dict]]) -> list[Finding]:
    """Convert pipeline JSON predictions into ORM rows (AI-sourced)."""
    out: list[Finding] = []

    for row in predictions.get("quadrants", []):
        qid = int(row["quadrant_id"])
        tooth_label = f"Q{qid + 1}"
        box = row.get("box_xyxy") or []
        poly = _box_to_polygon(box) if len(box) == 4 else []
        out.append(
            Finding(
                analysis_id=analysis_id,
                tooth_label=tooth_label,
                finding_class="Quadrant Region",
                layer=FindingLayer.QUADRANTS,
                confidence=1.0,
                box_xyxy=[float(x) for x in box] if box else None,
                polygon=poly,
                accepted=True,
                source=FindingSource.AI,
            )
        )

    for row in predictions.get("teeth", []):
        label = str(row.get("label", "FDI-00"))
        conf = float(row.get("confidence", 0.0))
        box = row.get("box_xyxy") or []
        poly = _mask_to_polygon(row.get("mask_xy"))
        out.append(
            Finding(
                analysis_id=analysis_id,
                tooth_label=label,
                finding_class="AI Tooth Mask",
                layer=FindingLayer.TEETH,
                confidence=conf,
                box_xyxy=[float(x) for x in box] if box else None,
                polygon=poly,
                accepted=True,
                source=FindingSource.AI,
            )
        )

    for row in predictions.get("periapical", []):
        label = str(row.get("label", f"P{row.get('class_id', 0)}"))
        conf = float(row.get("confidence", 0.0))
        box = row.get("box_xyxy") or []
        poly = _mask_to_polygon(row.get("mask_xy"))
        out.append(
            Finding(
                analysis_id=analysis_id,
                tooth_label=label,
                finding_class="Periapical Lesion",
                layer=FindingLayer.PERIAPICAL,
                confidence=conf,
                box_xyxy=[float(x) for x in box] if box else None,
                polygon=poly,
                accepted=True,
                source=FindingSource.AI,
            )
        )

    for row in predictions.get("teeth_classification", []):
        qid = int(row.get("quadrant_id", 0))
        cls_name = str(row.get("label", "Finding"))
        conf = float(row.get("confidence", 0.0))
        box = row.get("box_xyxy") or []
        poly = _mask_to_polygon(row.get("mask_xy"))
        tooth_label = f"Q{qid + 1}-{cls_name}"
        out.append(
            Finding(
                analysis_id=analysis_id,
                tooth_label=tooth_label,
                finding_class=cls_name,
                layer=FindingLayer.TEETH,
                confidence=conf,
                box_xyxy=[float(x) for x in box] if box else None,
                polygon=poly,
                accepted=True,
                source=FindingSource.AI,
            )
        )

    return out


def _predictions_to_guest_findings(
    analysis_id: uuid.UUID, predictions: dict[str, list[dict]]
) -> list[GuestFinding]:
    """Convert pipeline predictions into in-memory guest findings."""
    ts = datetime.now(timezone.utc)
    out: list[GuestFinding] = []

    for row in predictions.get("quadrants", []):
        qid = int(row["quadrant_id"])
        tooth_label = f"Q{qid + 1}"
        box = row.get("box_xyxy") or []
        poly = _box_to_polygon(box) if len(box) == 4 else []
        bx = [float(x) for x in box] if box else None
        out.append(
            GuestFinding(
                id=uuid.uuid4(),
                analysis_id=analysis_id,
                tooth_label=tooth_label,
                finding_class="Quadrant Region",
                layer=FindingLayer.QUADRANTS,
                confidence=1.0,
                box_xyxy=bx,
                polygon=poly,
                accepted=True,
                source=FindingSource.AI,
                created_at=ts,
                updated_at=ts,
            )
        )

    for row in predictions.get("teeth", []):
        label = str(row.get("label", "FDI-00"))
        conf = float(row.get("confidence", 0.0))
        box = row.get("box_xyxy") or []
        poly = _mask_to_polygon(row.get("mask_xy"))
        bx = [float(x) for x in box] if box else None
        out.append(
            GuestFinding(
                id=uuid.uuid4(),
                analysis_id=analysis_id,
                tooth_label=label,
                finding_class="AI Tooth Mask",
                layer=FindingLayer.TEETH,
                confidence=conf,
                box_xyxy=bx,
                polygon=poly,
                accepted=True,
                source=FindingSource.AI,
                created_at=ts,
                updated_at=ts,
            )
        )

    for row in predictions.get("periapical", []):
        label = str(row.get("label", f"P{row.get('class_id', 0)}"))
        conf = float(row.get("confidence", 0.0))
        box = row.get("box_xyxy") or []
        poly = _mask_to_polygon(row.get("mask_xy"))
        bx = [float(x) for x in box] if box else None
        out.append(
            GuestFinding(
                id=uuid.uuid4(),
                analysis_id=analysis_id,
                tooth_label=label,
                finding_class="Periapical Lesion",
                layer=FindingLayer.PERIAPICAL,
                confidence=conf,
                box_xyxy=bx,
                polygon=poly,
                accepted=True,
                source=FindingSource.AI,
                created_at=ts,
                updated_at=ts,
            )
        )

    for row in predictions.get("teeth_classification", []):
        qid = int(row.get("quadrant_id", 0))
        cls_name = str(row.get("label", "Finding"))
        conf = float(row.get("confidence", 0.0))
        box = row.get("box_xyxy") or []
        poly = _mask_to_polygon(row.get("mask_xy"))
        tooth_label = f"Q{qid + 1}-{cls_name}"
        bx = [float(x) for x in box] if box else None
        out.append(
            GuestFinding(
                id=uuid.uuid4(),
                analysis_id=analysis_id,
                tooth_label=tooth_label,
                finding_class=cls_name,
                layer=FindingLayer.TEETH,
                confidence=conf,
                box_xyxy=bx,
                polygon=poly,
                accepted=True,
                source=FindingSource.AI,
                created_at=ts,
                updated_at=ts,
            )
        )

    return out


@dataclass(frozen=True)
class _MaterializedPipelineResult:
    """Pipeline predictions plus artifact bytes (paths are invalid after temp dir teardown)."""

    predictions: dict[str, list[dict]]
    file_bytes_by_key: dict[str, bytes]


async def _run_pipeline_from_bytes(
    filename: str, original_bytes: bytes, model_registry: Any
) -> _MaterializedPipelineResult:
    """Run modeling pipeline on in-memory image bytes.

    ``TemporaryDirectory`` is torn down as soon as the ``with`` block ends. We read all
    artifact files into memory *before* exit so callers are not left with dead ``Path``s.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir) / filename
        tmp_path.write_bytes(original_bytes)
        out_dir = Path(tmp_dir) / "out"
        out_dir.mkdir(parents=True, exist_ok=True)
        pipeline_output = await asyncio.to_thread(_run_pipeline_sync, tmp_path, out_dir, model_registry)
        file_bytes = {key: Path(path).read_bytes() for key, path in pipeline_output.files.items()}
    return _MaterializedPipelineResult(
        predictions=pipeline_output.predictions,
        file_bytes_by_key=file_bytes,
    )


def _run_pipeline_sync(
    image_path: Path,
    output_dir: Path,
    model_registry: Any,
) -> Any:
    """Import and execute pipeline in worker thread (blocking)."""
    setup_modeling_path()
    from utils.pipeline import run_pipeline

    settings = get_settings()
    return run_pipeline(
        image_path,
        tasks=["all"],
        output_dir=output_dir,
        conf_quadrants=settings.conf_quadrants,
        conf_teeth=settings.conf_teeth,
        conf_periapical=settings.conf_periapical,
        conf_teeth_classification=settings.conf_teeth_classification,
        model_registry=model_registry,
        parallel_crop_models=settings.inference_parallel_crop_models,
    )


async def process_analysis_inference(analysis_id: uuid.UUID, model_registry: Any) -> None:
    """Load stored image, run YOLO pipeline, persist overlays and findings."""
    async with AsyncSessionLocal() as session:
        stmt = (
            select(Analysis)
            .where(Analysis.id == analysis_id)
            .options()
        )
        result = await session.execute(stmt)
        analysis = result.scalar_one_or_none()
        if analysis is None:
            return

        img_stmt = select(ImageAsset).where(
            ImageAsset.analysis_id == analysis_id,
            ImageAsset.kind == ImageAssetKind.ORIGINAL,
        )
        img_res = await session.execute(img_stmt)
        original_asset = img_res.scalar_one_or_none()
        if original_asset is None:
            analysis.status = AnalysisStatus.FAILED
            analysis.error = "Original image asset missing."
            await session.commit()
            return

        try:
            pipeline_output = await _run_pipeline_from_bytes(
                analysis.filename, original_asset.data, model_registry
            )

            await session.execute(
                delete(Finding).where(
                    Finding.analysis_id == analysis_id,
                    Finding.source == FindingSource.AI,
                )
            )

            new_findings = _predictions_to_findings(analysis_id, pipeline_output.predictions)
            for f in new_findings:
                session.add(f)

            for key, data in pipeline_output.file_bytes_by_key.items():
                if key == "predictions_json":
                    continue
                kind = _PIPELINE_FILE_TO_KIND.get(key)
                if kind is None:
                    continue
                existing_stmt = select(ImageAsset).where(
                    ImageAsset.analysis_id == analysis_id,
                    ImageAsset.kind == kind,
                )
                existing = (await session.execute(existing_stmt)).scalar_one_or_none()
                mime, w, h = mime_and_dimensions(data)
                if existing:
                    existing.data = data
                    existing.mime_type = mime
                    existing.width = w
                    existing.height = h
                elif kind == ImageAssetKind.ORIGINAL:
                    pass
                else:
                    session.add(
                        ImageAsset(
                            analysis_id=analysis_id,
                            kind=kind,
                            mime_type=mime,
                            data=data,
                            width=w,
                            height=h,
                        )
                    )

            analysis.alert_level = _compute_alert_level(pipeline_output.predictions)
            analysis.status = AnalysisStatus.REVIEWING
            analysis.completed_at = datetime.now(timezone.utc)
            analysis.error = None

            await append_system_audit(
                session,
                analysis_id=analysis_id,
                reviewer="system",
                action="Inference completed",
                action_type=AuditActionType.SYSTEM,
            )
            await session.commit()
        except Exception as exc:  # noqa: BLE001 — capture for audit + UI
            # Clear any aborted transaction from earlier flush/insert failures
            # so we can still persist a reliable FAILED status.
            await session.rollback()
            analysis = await session.get(Analysis, analysis_id)
            if analysis is None:
                return
            analysis.status = AnalysisStatus.FAILED
            analysis.error = str(exc)
            try:
                await append_system_audit(
                    session,
                    analysis_id=analysis_id,
                    reviewer="system",
                    action=f"Inference failed: {exc}",
                    action_type=AuditActionType.SYSTEM,
                )
                await session.commit()
            except Exception:
                await session.rollback()
                analysis = await session.get(Analysis, analysis_id)
                if analysis is None:
                    return
                analysis.status = AnalysisStatus.FAILED
                analysis.error = str(exc)
                await session.commit()


async def process_guest_inference(
    analysis_id: uuid.UUID,
    model_registry: Any,
    workspace: GuestWorkspace,
) -> None:
    """Run pipeline for a guest analysis and write results into ``GuestWorkspace``."""
    record = workspace.get(analysis_id)
    if record is None:
        return
    original = record.assets.get(ImageAssetKind.ORIGINAL)
    if original is None:
        record.status = AnalysisStatus.FAILED
        record.error = "Original image asset missing."
        workspace.append_audit(
            record,
            reviewer="system",
            action="Original image asset missing.",
            action_type=AuditActionType.SYSTEM,
        )
        return

    try:
        pipeline_output = await _run_pipeline_from_bytes(record.filename, original.data, model_registry)
        record.findings = [f for f in record.findings if f.source != FindingSource.AI]
        new_findings = _predictions_to_guest_findings(analysis_id, pipeline_output.predictions)
        workspace.extend_findings(record, new_findings)

        for key, data in pipeline_output.file_bytes_by_key.items():
            if key == "predictions_json":
                continue
            kind = _PIPELINE_FILE_TO_KIND.get(key)
            if kind is None or kind == ImageAssetKind.ORIGINAL:
                continue
            mime, w, h = mime_and_dimensions(data)
            record.assets[kind] = GuestImageAsset(kind=kind, mime_type=mime, data=data, width=w, height=h)

        record.alert_level = _compute_alert_level(pipeline_output.predictions)
        record.status = AnalysisStatus.REVIEWING
        record.completed_at = datetime.now(timezone.utc)
        record.error = None
        workspace.append_audit(
            record,
            reviewer="system",
            action="Inference completed",
            action_type=AuditActionType.SYSTEM,
        )
    except Exception as exc:  # noqa: BLE001
        record.status = AnalysisStatus.FAILED
        record.error = str(exc)
        workspace.append_audit(
            record,
            reviewer="system",
            action=f"Inference failed: {exc}",
            action_type=AuditActionType.SYSTEM,
        )

