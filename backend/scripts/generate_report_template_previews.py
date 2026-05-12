"""Generate three sample PDFs (one per layout) under ``backend/report_template_samples/``.

Run from the repository root::

    python -m backend.scripts.generate_report_template_previews

Open the PDFs in any viewer to compare formats before wiring a template into production.
"""

from __future__ import annotations

import struct
import uuid
import zlib
from datetime import datetime, timezone
from pathlib import Path

from backend.app.schemas.analysis import ReportFindingPreview, ReportPreviewResponse
from backend.app.services.report_pdf_templates import (
    REPORT_PDF_TEMPLATE_LABELS,
    ReportPdfTemplateId,
    render_report_pdf_with_template,
)


def _solid_png(width: int, height: int, rgb: tuple[int, int, int]) -> bytes:
    """Build a minimal truecolor PNG without Pillow (placeholder radiograph)."""

    def _chunk(chunk_type: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(chunk_type + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + chunk_type + data + struct.pack(">I", crc)

    r, g, b = rgb
    row = b"\x00" + bytes([r, g, b] * width)
    raw = row * height
    compressed = zlib.compress(raw, 9)
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", ihdr) + _chunk(b"IDAT", compressed) + _chunk(b"IEND", b"")


def _sample_preview(analysis_id: uuid.UUID) -> ReportPreviewResponse:
    return ReportPreviewResponse(
        analysis_id=analysis_id,
        patient_name="Sample Patient",
        patient_code="P-DEMO-1042",
        scan_date=datetime(2026, 4, 18, 10, 30, tzinfo=timezone.utc),
        accepted_findings_count=4,
        accepted_findings=[
            ReportFindingPreview(
                id=uuid.uuid4(),
                tooth_label="16",
                finding="Caries",
                confidence=0.82,
                layer="teeth_overlay",
            ),
            ReportFindingPreview(
                id=uuid.uuid4(),
                tooth_label="36",
                finding="Deep Caries",
                confidence=0.71,
                layer="teeth_overlay",
            ),
            ReportFindingPreview(
                id=uuid.uuid4(),
                tooth_label="46",
                finding="Periapical Lesion",
                confidence=0.64,
                layer="periapical",
            ),
            ReportFindingPreview(
                id=uuid.uuid4(),
                tooth_label="38",
                finding="Impacted",
                confidence=0.59,
                layer="teeth_overlay",
            ),
        ],
        sections=[
            {
                "title": "Clinical summary",
                "body": (
                    "Panoramic radiograph reviewed with AI-assisted overlays. "
                    "Posterior quadrants show restorations and focal radiolucencies "
                    "consistent with the listed detections pending clinical correlation."
                ),
            },
            {
                "title": "Impression",
                "body": (
                    "Findings suggest active carious lesions and a periapical change "
                    "in the mandibular right molar region, plus an impacted third molar."
                ),
            },
            {
                "title": "Recommendations",
                "body": (
                    "Correlate with clinical exam and vitality testing; consider "
                    "targeted intraoral imaging where treatment is planned."
                ),
            },
            {
                "title": "Reviewer confirmation",
                "body": "I reviewed AI detections and accept the listed findings for this export.",
            },
        ],
        image_kinds=["original", "quadrants_overlay", "teeth_overlay", "periapical_quadrants_overlay"],
    )


def _sample_evidence() -> list[tuple[str, bytes]]:
    return [
        ("quadrants_overlay", _solid_png(640, 320, (30, 58, 95))),
        ("teeth_overlay", _solid_png(640, 320, (55, 48, 120))),
    ]


def _sample_pathology_crops() -> dict[str, list[dict[str, object]]]:
    return {
        "Caries Findings": [
            {
                "tooth": "16",
                "finding": "Caries",
                "confidence": 0.82,
                "image": _solid_png(280, 200, (120, 40, 40)),
            },
            {
                "tooth": "36",
                "finding": "Deep Caries",
                "confidence": 0.71,
                "image": _solid_png(280, 200, (140, 55, 30)),
            },
        ],
        "Periapical Findings": [
            {
                "tooth": "46",
                "finding": "Periapical Lesion",
                "confidence": 0.64,
                "image": _solid_png(280, 200, (40, 90, 120)),
            },
        ],
        "Impacted Findings": [
            {
                "tooth": "38",
                "finding": "Impacted",
                "confidence": 0.59,
                "image": _solid_png(280, 200, (70, 70, 90)),
            },
        ],
    }


def main() -> None:
    repo_backend = Path(__file__).resolve().parent.parent
    out_dir = repo_backend / "report_template_samples"
    out_dir.mkdir(parents=True, exist_ok=True)

    analysis_id = uuid.UUID("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")
    preview = _sample_preview(analysis_id)
    evidence = _sample_evidence()
    pathology = _sample_pathology_crops()
    fixed_time = datetime(2026, 5, 12, 14, 0, tzinfo=timezone.utc)

    for tid in REPORT_PDF_TEMPLATE_LABELS:
        template_id: ReportPdfTemplateId = tid  # type: ignore[assignment]
        pdf_bytes = render_report_pdf_with_template(
            template_id,
            preview=preview,
            reviewer="Dr. A. Example",
            evidence_images=evidence,
            pathology_crops=pathology,
            generated_at=fixed_time,
        )
        path = out_dir / f"toothfairy_report_template_{tid}.pdf"
        path.write_bytes(pdf_bytes)
        print(f"Wrote {path} — {REPORT_PDF_TEMPLATE_LABELS[template_id]}")

    print("\nOpen these three files side by side to pick a default layout for production.")


if __name__ == "__main__":
    main()
