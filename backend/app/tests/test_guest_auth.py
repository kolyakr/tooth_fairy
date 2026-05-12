"""Authorization and guest (non-DB) analysis behavior."""

from __future__ import annotations

import asyncio
import base64
import io

from sqlalchemy import func, select

from backend.app.core import database as database_module
from backend.app.db.models import Analysis

_MIN_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


async def _analysis_count() -> int:
    async with database_module.AsyncSessionLocal() as session:
        return int((await session.execute(select(func.count()).select_from(Analysis))).scalar_one())


def test_guest_upload_skips_database(guest_client):
    """Guest multipart upload does not insert into ``analyses`` table."""
    n0 = asyncio.run(_analysis_count())

    files = {"file": ("tiny.png", io.BytesIO(_MIN_PNG), "image/png")}
    data = {
        "patient_code": "P-GUEST",
        "patient_name": "Guest",
        "age": "40",
        "scan_date": "2026-05-01",
        "chief_complaint": "demo",
    }
    res = guest_client.post("/api/v1/analyses", files=files, data=data)
    assert res.status_code == 200, res.text
    body = res.json()
    aid = body["id"]

    n1 = asyncio.run(_analysis_count())
    assert n0 == n1

    cookie = res.headers.get("set-cookie", "")
    assert "toothfairy_guest=" in cookie

    cookie_header = cookie.split(";")[0].strip()
    lst = guest_client.get("/api/v1/analyses?limit=10", headers={"Cookie": cookie_header})
    assert lst.status_code == 200
    rows = lst.json()
    assert len(rows) == 1
    assert rows[0]["id"] == aid

    detail = guest_client.get(f"/api/v1/analyses/{aid}", headers={"Cookie": cookie_header})
    assert detail.status_code == 200
    assert detail.json()["status"] == "Reviewing"


def test_patients_requires_auth(guest_client):
    """``POST /patients`` returns 401 without Bearer token."""
    res = guest_client.post(
        "/api/v1/patients",
        json={"patient_code": "P-X", "name": "N", "age": 30},
    )
    assert res.status_code == 401
