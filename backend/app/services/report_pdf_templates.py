"""Alternative PDF layouts for ToothFairy radiology reports (design options).

Three distinct visual systems share the same data contract as ``render_report_pdf``
in ``report_service.py``. Use ``render_report_pdf_with_template`` to render, or run
``python -m backend.scripts.generate_report_template_previews`` for sample files.
"""

from __future__ import annotations

import io
from datetime import datetime, timezone
from typing import Any, Literal
from xml.sax.saxutils import escape as xml_escape

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Image as ReportLabImage
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from backend.app.schemas.analysis import ReportPreviewResponse

ReportPdfTemplateId = Literal["A_minimal", "B_letterhead", "C_evidence_atlas"]

REPORT_PDF_TEMPLATE_LABELS: dict[ReportPdfTemplateId, str] = {
    "A_minimal": "A — Minimal clinical (clean typography, list-style findings)",
    "B_letterhead": "B — Practice letterhead (branded header band, formal tables)",
    "C_evidence_atlas": "C — Evidence-first atlas (imaging grid, narrative after visuals)",
}


def _p(text: str) -> str:
    """Escape text for ReportLab Paragraph and preserve line breaks."""
    return xml_escape(text).replace("\n", "<br/>")


def _scale_image(image_bytes: bytes, max_w_mm: float, max_h_mm: float) -> ReportLabImage:
    max_w = max_w_mm * mm
    max_h = max_h_mm * mm
    pil = PILImage.open(io.BytesIO(image_bytes))
    iw, ih = pil.size
    iw = max(iw, 1)
    ih = max(ih, 1)
    ratio = min(max_w / float(iw), max_h / float(ih))
    draw_w = float(iw) * ratio
    draw_h = float(ih) * ratio
    return ReportLabImage(io.BytesIO(image_bytes), width=draw_w, height=draw_h)


def _pathology_story(
    pathology_crops: dict[str, list[dict[str, Any]]] | None,
    body: ParagraphStyle,
    muted: ParagraphStyle,
    h2: ParagraphStyle,
    max_crop_w_mm: float,
    max_crop_h_mm: float,
) -> list[Any]:
    story: list[Any] = []
    if not pathology_crops:
        return story
    for section_name in ("Periapical Findings", "Impacted Findings", "Caries Findings"):
        rows = pathology_crops.get(section_name, [])
        if not rows:
            continue
        story.append(Spacer(1, 2 * mm))
        story.append(Paragraph(_p(section_name), h2))
        story.append(Paragraph(_p(f"Total findings: {len(rows)}"), muted))
        story.append(Spacer(1, 2 * mm))
        for idx, item in enumerate(rows, start=1):
            story.append(
                Paragraph(
                    _p(
                        f"{idx}. {item['finding']} · {item['tooth']} · "
                        f"{int(round(float(item['confidence']) * 100))}%"
                    ),
                    body,
                )
            )
            if item.get("image"):
                story.append(_scale_image(item["image"], max_crop_w_mm, max_crop_h_mm))
            else:
                story.append(Paragraph(_p("Crop unavailable for this finding."), muted))
            story.append(Spacer(1, 2 * mm))
    return story


def render_template_a_minimal(
    *,
    preview: ReportPreviewResponse,
    reviewer: str,
    evidence_images: list[tuple[str, bytes]] | None = None,
    pathology_crops: dict[str, list[dict[str, Any]]] | None = None,
    generated_at: datetime | None = None,
) -> bytes:
    """Sparse layout: emphasis on reading comfort; findings as a numbered list."""
    timestamp = generated_at or datetime.now(timezone.utc)
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=22 * mm,
        rightMargin=22 * mm,
        topMargin=18 * mm,
        bottomMargin=20 * mm,
        title=f"ToothFairy report {preview.analysis_id}",
        author=reviewer,
    )
    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        "tm_title",
        parent=styles["Title"],
        fontSize=20,
        leading=24,
        spaceAfter=2 * mm,
        textColor=colors.HexColor("#0a0a0a"),
        fontName="Helvetica-Bold",
    )
    sub = ParagraphStyle(
        "tm_sub",
        parent=styles["BodyText"],
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#525252"),
    )
    h2 = ParagraphStyle(
        "tm_h2",
        parent=styles["Heading2"],
        fontSize=11,
        spaceBefore=5 * mm,
        spaceAfter=3 * mm,
        textColor=colors.HexColor("#171717"),
        fontName="Helvetica-Bold",
    )
    body = ParagraphStyle(
        "tm_body",
        parent=styles["BodyText"],
        fontSize=10,
        leading=15,
        textColor=colors.HexColor("#262626"),
    )
    story: list[Any] = []
    story.append(Paragraph(_p("Radiology report"), title))
    story.append(
        Paragraph(
            _p(
                f"{preview.patient_name} · {preview.patient_code} · "
                f"Scan {preview.scan_date or 'N/A'} · Reviewer {reviewer}"
            ),
            sub,
        )
    )
    story.append(Paragraph(_p(f"Analysis {preview.analysis_id} · {timestamp.isoformat()}"), sub))
    story.append(Spacer(1, 8 * mm))

    story.append(Paragraph(_p("Overview"), h2))
    pathology_counts = {"Periapical Lesion": 0, "Impacted": 0, "Caries": 0}
    for row in preview.accepted_findings:
        if row.finding in pathology_counts:
            pathology_counts[row.finding] += 1
    overview = (
        f"Accepted AI-assisted findings: {preview.accepted_findings_count}. "
        f"Caries {pathology_counts['Caries']}, impacted {pathology_counts['Impacted']}, "
        f"periapical {pathology_counts['Periapical Lesion']}."
    )
    story.append(Paragraph(_p(overview), body))

    story.append(Paragraph(_p("Tooth-level summary"), h2))
    if preview.accepted_findings:
        for i, row in enumerate(preview.accepted_findings, start=1):
            line = (
                f"{i}. {row.tooth_label} — {row.finding} "
                f"({int(round(row.confidence * 100))}%, {row.layer})"
            )
            story.append(Paragraph(_p(line), body))
    else:
        story.append(Paragraph(_p("No accepted findings."), body))

    for section in preview.sections:
        story.append(Paragraph(_p(section["title"]), h2))
        story.append(Paragraph(_p(section["body"]), body))

    if preview.image_kinds:
        story.append(Paragraph(_p("Source overlays"), h2))
        story.append(Paragraph(_p(", ".join(preview.image_kinds)), body))

    if evidence_images:
        story.append(Paragraph(_p("Attached figures"), h2))
        for kind, image_bytes in evidence_images:
            story.append(Paragraph(_p(kind.replace("_", " ").title()), sub))
            story.append(_scale_image(image_bytes, 160, 78))
            story.append(Spacer(1, 3 * mm))

    story.extend(_pathology_story(pathology_crops, body, sub, h2, 115, 58))

    story.append(Spacer(1, 10 * mm))
    story.append(
        Paragraph(
            _p("AI assists interpretation; clinical decisions remain with the reviewing dentist."),
            sub,
        )
    )

    def _footer(canvas: Any, _doc: Any) -> None:
        w, _h = A4
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#e5e5e5"))
        canvas.setLineWidth(0.3)
        canvas.line(22 * mm, 14 * mm, w - 22 * mm, 14 * mm)
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#737373"))
        canvas.drawString(22 * mm, 10 * mm, str(preview.analysis_id))
        canvas.drawRightString(w - 22 * mm, 10 * mm, f"Page {canvas.getPageNumber()}")
        canvas.restoreState()

    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    return buf.getvalue()


def render_template_b_letterhead(
    *,
    preview: ReportPreviewResponse,
    reviewer: str,
    evidence_images: list[tuple[str, bytes]] | None = None,
    pathology_crops: dict[str, list[dict[str, Any]]] | None = None,
    generated_at: datetime | None = None,
) -> bytes:
    """Formal clinic letter: branded header band, metadata grid, tabular findings.

    This is the production PDF layout (``report_service.render_report_pdf`` delegates here).
    """
    timestamp = generated_at or datetime.now(timezone.utc)
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=26 * mm,
        bottomMargin=16 * mm,
        title=f"ToothFairy report {preview.analysis_id}",
        author=reviewer,
    )
    styles = getSampleStyleSheet()
    h1 = ParagraphStyle(
        "tl_h1",
        parent=styles["Heading1"],
        fontSize=14,
        spaceAfter=4,
        textColor=colors.HexColor("#1e293b"),
    )
    h2 = ParagraphStyle(
        "tl_h2",
        parent=styles["Heading2"],
        fontSize=11,
        spaceAfter=4,
        textColor=colors.HexColor("#0f766e"),
    )
    body = ParagraphStyle("tl_body", parent=styles["BodyText"], fontSize=10, leading=14)
    muted = ParagraphStyle("tl_muted", parent=body, textColor=colors.HexColor("#64748b"))
    box = ParagraphStyle("tl_box", parent=body, backColor=colors.HexColor("#f1f5f9"), borderPadding=8)

    story: list[Any] = []
    story.append(Paragraph(_p("Panoramic radiograph — structured report"), h1))
    story.append(Spacer(1, 2 * mm))

    meta = Table(
        [
            [
                Paragraph(_p("Patient"), muted),
                Paragraph(_p(preview.patient_name), body),
                Paragraph(_p("Patient ID"), muted),
                Paragraph(_p(preview.patient_code), body),
            ],
            [
                Paragraph(_p("Scan date"), muted),
                Paragraph(_p(str(preview.scan_date or "N/A")), body),
                Paragraph(_p("Reviewer"), muted),
                Paragraph(_p(reviewer), body),
            ],
            [
                Paragraph(_p("Analysis ID"), muted),
                Paragraph(_p(str(preview.analysis_id)), body),
                Paragraph(_p("Generated (UTC)"), muted),
                Paragraph(_p(timestamp.isoformat()), body),
            ],
        ],
        colWidths=[28 * mm, 58 * mm, 32 * mm, 58 * mm],
    )
    meta.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#cbd5e1")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f8fafc")),
                ("BACKGROUND", (2, 0), (2, -1), colors.HexColor("#f8fafc")),
            ]
        )
    )
    story.append(meta)
    story.append(Spacer(1, 5 * mm))

    pathology_counts = {"Periapical Lesion": 0, "Impacted": 0, "Caries": 0}
    for row in preview.accepted_findings:
        if row.finding in pathology_counts:
            pathology_counts[row.finding] += 1
    story.append(Paragraph(_p("Quantitative summary"), h2))
    story.append(
        Paragraph(
            _p(
                f"Accepted findings: {preview.accepted_findings_count}. "
                f"Caries: {pathology_counts['Caries']}; "
                f"Impacted: {pathology_counts['Impacted']}; "
                f"Periapical: {pathology_counts['Periapical Lesion']}."
            ),
            box,
        )
    )
    story.append(Spacer(1, 4 * mm))

    story.append(Paragraph(_p("Confirmed findings (tabular)"), h2))
    if preview.accepted_findings:
        data: list[list[Any]] = [["Tooth", "Finding", "Confidence", "Layer"]]
        for row in preview.accepted_findings:
            data.append(
                [
                    row.tooth_label,
                    row.finding,
                    f"{int(round(row.confidence * 100))}%",
                    row.layer,
                ]
            )
        table = Table(data, repeatRows=1)
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#ccfbf1")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#134e4a")),
                    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#94a3b8")),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#fafafa")]),
                ]
            )
        )
        story.append(table)
    else:
        story.append(Paragraph(_p("No accepted findings recorded."), body))
    story.append(Spacer(1, 4 * mm))

    for section in preview.sections:
        story.append(Paragraph(_p(section["title"]), h2))
        story.append(Paragraph(_p(section["body"]), box))
        story.append(Spacer(1, 3 * mm))

    if preview.image_kinds:
        story.append(Paragraph(_p("Image kinds on file"), h2))
        story.append(Paragraph(_p(", ".join(preview.image_kinds)), body))

    if evidence_images:
        story.append(Spacer(1, 3 * mm))
        story.append(Paragraph(_p("Embedded radiographic figures"), h2))
        for kind, image_bytes in evidence_images:
            story.append(Paragraph(_p(kind.replace("_", " ").title()), muted))
            story.append(_scale_image(image_bytes, 168, 82))
            story.append(Spacer(1, 3 * mm))

    story.extend(_pathology_story(pathology_crops, body, muted, h2, 120, 60))

    story.append(Spacer(1, 6 * mm))
    story.append(
        Paragraph(
            _p(
                "Disclaimer: This document summarizes AI-assisted detections that were "
                "reviewed by the named clinician. It is not a standalone diagnosis."
            ),
            muted,
        )
    )

    def _letterhead(canvas: Any, _doc: Any) -> None:
        w, h = A4
        canvas.saveState()
        canvas.setFillColor(colors.HexColor("#0f766e"))
        canvas.rect(0, h - 18 * mm, w, 18 * mm, fill=1, stroke=0)
        canvas.setFillColor(colors.white)
        canvas.setFont("Helvetica-Bold", 12)
        canvas.drawString(18 * mm, h - 11 * mm, "TOOTHFAIRY — DENTAL IMAGING REPORT")
        canvas.setFont("Helvetica", 8)
        canvas.drawRightString(w - 18 * mm, h - 11 * mm, "CONFIDENTIAL")
        canvas.setStrokeColor(colors.HexColor("#e2e8f0"))
        canvas.line(18 * mm, 15 * mm, w - 18 * mm, 15 * mm)
        canvas.setFillColor(colors.HexColor("#64748b"))
        canvas.setFont("Helvetica", 8)
        canvas.drawString(18 * mm, 10 * mm, f"Ref: {preview.analysis_id}")
        canvas.drawRightString(w - 18 * mm, 10 * mm, f"Page {canvas.getPageNumber()}")
        canvas.restoreState()

    doc.build(story, onFirstPage=_letterhead, onLaterPages=_letterhead)
    return buf.getvalue()


def render_template_c_evidence_atlas(
    *,
    preview: ReportPreviewResponse,
    reviewer: str,
    evidence_images: list[tuple[str, bytes]] | None = None,
    pathology_crops: dict[str, list[dict[str, Any]]] | None = None,
    generated_at: datetime | None = None,
) -> bytes:
    """Atlas style: imaging grid first, then dense clinical text and crop montages."""
    timestamp = generated_at or datetime.now(timezone.utc)
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=16 * mm,
        rightMargin=16 * mm,
        topMargin=14 * mm,
        bottomMargin=16 * mm,
        title=f"ToothFairy report {preview.analysis_id}",
        author=reviewer,
    )
    styles = getSampleStyleSheet()
    banner = ParagraphStyle(
        "ta_banner",
        parent=styles["Normal"],
        fontSize=9,
        textColor=colors.white,
        backColor=colors.HexColor("#312e81"),
        borderPadding=6,
        leading=12,
    )
    h2 = ParagraphStyle(
        "ta_h2",
        parent=styles["Heading2"],
        fontSize=11,
        spaceAfter=4,
        textColor=colors.HexColor("#312e81"),
    )
    body = ParagraphStyle("ta_body", parent=styles["BodyText"], fontSize=9, leading=12)
    muted = ParagraphStyle("ta_muted", parent=body, textColor=colors.HexColor("#6b7280"))
    caption = ParagraphStyle("ta_cap", parent=muted, fontSize=8, leading=10)

    story: list[Any] = []
    story.append(
        Paragraph(
            _p(
                f"EVIDENCE ATLAS · {preview.patient_name} ({preview.patient_code}) · "
                f"{preview.scan_date or 'N/A'}"
            ),
            banner,
        )
    )
    story.append(Spacer(1, 2 * mm))
    story.append(
        Paragraph(
            _p(f"Reviewer: {reviewer} · Analysis: {preview.analysis_id} · {timestamp.isoformat()}"),
            muted,
        )
    )
    story.append(Spacer(1, 4 * mm))

    # Imaging grid (two figures per row; empty cell if odd count)
    if evidence_images:
        story.append(Paragraph(_p("Figure panel"), h2))
        row_cells: list[list[Any]] = []
        cells: list[Any] = []
        for kind, image_bytes in evidence_images:
            inner = Table(
                [
                    [_scale_image(image_bytes, 78, 52)],
                    [Paragraph(_p(kind.replace("_", " ").title()), caption)],
                ],
                colWidths=[76 * mm],
            )
            inner.setStyle(
                TableStyle(
                    [
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("LEFTPADDING", (0, 0), (-1, -1), 0),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                    ]
                )
            )
            cells.append(inner)
        for i in range(0, len(cells), 2):
            left = cells[i]
            right = cells[i + 1] if i + 1 < len(cells) else Paragraph("", caption)
            row_cells.append([left, right])
        grid = Table(row_cells, colWidths=[82 * mm, 82 * mm])
        grid.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 2)]))
        story.append(grid)
        story.append(Spacer(1, 5 * mm))

    pathology_counts = {"Periapical Lesion": 0, "Impacted": 0, "Caries": 0}
    for row in preview.accepted_findings:
        if row.finding in pathology_counts:
            pathology_counts[row.finding] += 1
    story.append(Paragraph(_p("Case statistics"), h2))
    stats_tbl = Table(
        [
            [
                Paragraph(_p("Accepted"), body),
                Paragraph(_p(str(preview.accepted_findings_count)), body),
                Paragraph(_p("Caries"), body),
                Paragraph(_p(str(pathology_counts["Caries"])), body),
            ],
            [
                Paragraph(_p("Impacted"), body),
                Paragraph(_p(str(pathology_counts["Impacted"])), body),
                Paragraph(_p("Periapical"), body),
                Paragraph(_p(str(pathology_counts["Periapical Lesion"])), body),
            ],
        ],
        colWidths=[28 * mm, 32 * mm, 32 * mm, 32 * mm],
    )
    stats_tbl.setStyle(
        TableStyle(
            [
                ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#c7d2fe")),
                ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#e0e7ff")),
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#eef2ff")),
            ]
        )
    )
    story.append(stats_tbl)
    story.append(Spacer(1, 4 * mm))

    story.append(Paragraph(_p("Narrative sections"), h2))
    for section in preview.sections:
        story.append(Paragraph(_p(section["title"]), ParagraphStyle("ta_h3", parent=h2, fontSize=10)))
        story.append(Paragraph(_p(section["body"]), body))
        story.append(Spacer(1, 2 * mm))

    story.append(PageBreak())
    story.append(Paragraph(_p("Finding register"), h2))
    if preview.accepted_findings:
        reg_data: list[list[str]] = [["Tooth", "Finding", "Conf.", "Layer"]]
        for row in preview.accepted_findings:
            reg_data.append(
                [
                    row.tooth_label,
                    row.finding,
                    f"{int(round(row.confidence * 100))}%",
                    row.layer,
                ]
            )
        reg = Table(reg_data, repeatRows=1)
        reg.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#312e81")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#9ca3af")),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                ]
            )
        )
        story.append(reg)
    else:
        story.append(Paragraph(_p("No accepted findings."), body))

    if preview.image_kinds:
        story.append(Spacer(1, 3 * mm))
        story.append(Paragraph(_p("Layer keys"), h2))
        story.append(Paragraph(_p(", ".join(preview.image_kinds)), body))

    # Pathology montage: two crops per row when possible
    if pathology_crops:
        story.append(Spacer(1, 4 * mm))
        story.append(Paragraph(_p("Pathology crops"), h2))
        for section_name in ("Periapical Findings", "Impacted Findings", "Caries Findings"):
            rows = pathology_crops.get(section_name, [])
            if not rows:
                continue
            story.append(Paragraph(_p(f"{section_name} ({len(rows)})"), muted))
            montage_rows: list[list[Any]] = []
            buf_cells: list[Any] = []
            for item in rows:
                label = Paragraph(
                    _p(
                        f"{item['finding']} · {item['tooth']} · "
                        f"{int(round(float(item['confidence']) * 100))}%"
                    ),
                    caption,
                )
                if item.get("image"):
                    img = _scale_image(item["image"], 72, 44)
                    inner = Table([[label], [img]], colWidths=[76 * mm])
                else:
                    inner = Table(
                        [[label], [Paragraph(_p("(no crop)"), caption)]],
                        colWidths=[76 * mm],
                    )
                inner.setStyle(
                    TableStyle(
                        [
                            ("VALIGN", (0, 0), (-1, -1), "TOP"),
                            ("LEFTPADDING", (0, 0), (-1, -1), 0),
                            ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                        ]
                    )
                )
                buf_cells.append(inner)
            for i in range(0, len(buf_cells), 2):
                left = buf_cells[i]
                right = buf_cells[i + 1] if i + 1 < len(buf_cells) else Paragraph("", caption)
                montage_rows.append([left, right])
            if montage_rows:
                mt = Table(montage_rows, colWidths=[78 * mm, 78 * mm])
                mt.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")]))
                story.append(mt)
                story.append(Spacer(1, 3 * mm))

    story.append(Spacer(1, 6 * mm))
    story.append(
        Paragraph(
            _p("Human-in-the-loop: all listed findings were accepted by the reviewer before export."),
            muted,
        )
    )

    def _atlas_footer(canvas: Any, _doc: Any) -> None:
        w, h = A4
        canvas.saveState()
        canvas.setFillColor(colors.HexColor("#312e81"))
        canvas.rect(0, 0, w, 10 * mm, fill=1, stroke=0)
        canvas.setFillColor(colors.HexColor("#e0e7ff"))
        canvas.setFont("Helvetica", 7)
        canvas.drawString(16 * mm, 3.5 * mm, f"Atlas · {preview.analysis_id}")
        canvas.drawRightString(w - 16 * mm, 3.5 * mm, f"Page {canvas.getPageNumber()}")
        canvas.restoreState()

    doc.build(story, onFirstPage=_atlas_footer, onLaterPages=_atlas_footer)
    return buf.getvalue()


def render_report_pdf_with_template(
    template_id: ReportPdfTemplateId,
    *,
    preview: ReportPreviewResponse,
    reviewer: str,
    evidence_images: list[tuple[str, bytes]] | None = None,
    pathology_crops: dict[str, list[dict[str, Any]]] | None = None,
    generated_at: datetime | None = None,
) -> bytes:
    """Dispatch to the selected layout."""
    if template_id == "A_minimal":
        return render_template_a_minimal(
            preview=preview,
            reviewer=reviewer,
            evidence_images=evidence_images,
            pathology_crops=pathology_crops,
            generated_at=generated_at,
        )
    if template_id == "B_letterhead":
        return render_template_b_letterhead(
            preview=preview,
            reviewer=reviewer,
            evidence_images=evidence_images,
            pathology_crops=pathology_crops,
            generated_at=generated_at,
        )
    if template_id == "C_evidence_atlas":
        return render_template_c_evidence_atlas(
            preview=preview,
            reviewer=reviewer,
            evidence_images=evidence_images,
            pathology_crops=pathology_crops,
            generated_at=generated_at,
        )
    raise ValueError(f"Unknown template_id: {template_id!r}")
