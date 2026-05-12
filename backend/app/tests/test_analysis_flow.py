"""Smoke test: upload, background inference hook, review completion."""

from __future__ import annotations

import base64
import io

# 1x1 transparent PNG (valid structure)
_MIN_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


def test_upload_inference_review_flow(client):
    """Multipart upload schedules inference; detail reflects completion; review closes."""
    png = _MIN_PNG
    files = {"file": ("tiny.png", io.BytesIO(png), "image/png")}
    data = {
        "patient_code": "P-9999",
        "patient_name": "Test Patient",
        "age": "42",
        "scan_date": "2026-05-01",
        "chief_complaint": "pain",
    }
    res = client.post("/api/v1/analyses", files=files, data=data)
    assert res.status_code == 200, res.text
    aid = res.json()["id"]

    detail = client.get(f"/api/v1/analyses/{aid}")
    assert detail.status_code == 200
    body = detail.json()
    assert body["status"] == "Reviewing"
    assert body["findings_count"] >= 1

    complete = client.post(
        f"/api/v1/analyses/{aid}/complete-review",
        json={"reviewer": "Dr. Test"},
    )
    assert complete.status_code == 200
    assert complete.json()["status"] == "Reviewed"


def test_health(client):
    """Root health endpoint responds."""
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"


def test_report_generation_lifecycle(client):
    """Preview, generate, and download PDF report after review completion."""
    png = _MIN_PNG
    files = {"file": ("tiny.png", io.BytesIO(png), "image/png")}
    data = {
        "patient_code": "P-1000",
        "patient_name": "Report Patient",
        "age": "37",
        "scan_date": "2026-05-05",
        "chief_complaint": "sensitivity",
    }
    created = client.post("/api/v1/analyses", files=files, data=data)
    assert created.status_code == 200, created.text
    aid = created.json()["id"]

    draft = {
        "clinical_summary": "Panoramic scan reviewed with one accepted finding.",
        "impression": "Localized lesion suspicious for caries.",
        "recommendations": "Clinical exam and restoration planning recommended.",
        "reviewer_confirmation": "Reviewed and confirmed by Dr. Test",
        "include_images": True,
    }

    preview_pre = client.post(f"/api/v1/analyses/{aid}/report/preview", json=draft)
    assert preview_pre.status_code == 200, preview_pre.text
    assert preview_pre.json()["accepted_findings_count"] >= 1

    blocked = client.post(
        f"/api/v1/analyses/{aid}/report/generate",
        json={**draft, "reviewer": "Dr. Test"},
    )
    assert blocked.status_code == 422

    complete = client.post(
        f"/api/v1/analyses/{aid}/complete-review",
        json={"reviewer": "Dr. Test"},
    )
    assert complete.status_code == 200
    assert complete.json()["status"] == "Reviewed"

    generated = client.post(
        f"/api/v1/analyses/{aid}/report/generate",
        json={**draft, "reviewer": "Dr. Test"},
    )
    assert generated.status_code == 200, generated.text
    body = generated.json()
    assert body["status"] == "Report Generated"
    assert body["filename"].endswith(".pdf")
    assert body["download_url"].endswith("/report/download")

    detail = client.get(f"/api/v1/analyses/{aid}")
    assert detail.status_code == 200
    assert detail.json()["status"] == "Report Generated"

    download = client.get(f"/api/v1/analyses/{aid}/report/download")
    assert download.status_code == 200, download.text
    assert download.headers["content-type"].startswith("application/pdf")
    assert download.content.startswith(b"%PDF")
    assert len(download.content) > 1500
