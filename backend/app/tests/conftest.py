"""Pytest fixtures: in-memory SQLite and patched inference."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from backend.app.api.deps import AuthPrincipal, get_db, get_optional_principal
from backend.app.db.base import Base
from backend.app.db.models import (
    AlertLevel,
    Analysis,
    AnalysisStatus,
    Finding,
    FindingLayer,
    FindingSource,
)
from backend.app.main import app


@pytest.fixture(autouse=True)
def patch_model_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Avoid loading Ultralytics weights during app lifespan."""

    monkeypatch.setattr("backend.app.main.build_model_registry", lambda: object())


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """HTTP client against the ASGI app with isolated SQLite DB."""
    import asyncio

    engine = create_async_engine("sqlite+aiosqlite:///:memory:", future=True)

    async def _setup() -> None:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    asyncio.run(_setup())

    TestSession = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

    monkeypatch.setattr("backend.app.core.database.AsyncSessionLocal", TestSession)
    monkeypatch.setattr("backend.app.services.inference_service.AsyncSessionLocal", TestSession)

    async def override_db():
        async with TestSession() as session:
            yield session

    app.dependency_overrides[get_db] = override_db

    async def override_principal() -> AuthPrincipal:
        return AuthPrincipal(user_id="pytest")

    app.dependency_overrides[get_optional_principal] = override_principal

    async def fake_inference(analysis_id: uuid.UUID, registry: object) -> None:
        async with TestSession() as session:
            analysis = await session.get(Analysis, analysis_id)
            if analysis is None:
                return
            analysis.status = AnalysisStatus.REVIEWING
            analysis.alert_level = AlertLevel.LOW
            analysis.completed_at = datetime.now(timezone.utc)
            session.add(
                Finding(
                    analysis_id=analysis_id,
                    tooth_label="FDI-36",
                    finding_class="AI Tooth Mask",
                    layer=FindingLayer.TEETH,
                    confidence=0.9,
                    polygon=[
                        {"x": 10.0, "y": 10.0},
                        {"x": 20.0, "y": 10.0},
                        {"x": 15.0, "y": 20.0},
                    ],
                    accepted=True,
                    source=FindingSource.AI,
                )
            )
            await session.commit()

    monkeypatch.setattr(
        "backend.app.api.routes.analyses.process_analysis_inference",
        fake_inference,
    )

    with TestClient(app) as tc:
        yield tc

    app.dependency_overrides.clear()

    async def _dispose() -> None:
        await engine.dispose()

    asyncio.run(_dispose())


@pytest.fixture
def guest_client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """HTTP client without auth: guest (in-memory) analysis path."""
    import asyncio

    from backend.app.services.guest_workspace import GuestFinding, GuestWorkspace

    engine = create_async_engine("sqlite+aiosqlite:///:memory:", future=True)

    async def _setup() -> None:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    asyncio.run(_setup())

    TestSession = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

    monkeypatch.setattr("backend.app.core.database.AsyncSessionLocal", TestSession)
    monkeypatch.setattr("backend.app.services.inference_service.AsyncSessionLocal", TestSession)

    async def override_db():
        async with TestSession() as session:
            yield session

    app.dependency_overrides.clear()
    app.dependency_overrides[get_db] = override_db
    app.dependency_overrides.pop(get_optional_principal, None)

    async def fake_guest_inference(
        analysis_id: uuid.UUID,
        registry: object,
        workspace: GuestWorkspace,
    ) -> None:
        record = workspace.get(analysis_id)
        if record is None:
            return
        ts = datetime.now(timezone.utc)
        gf = GuestFinding(
            id=uuid.uuid4(),
            analysis_id=analysis_id,
            tooth_label="FDI-36",
            finding_class="AI Tooth Mask",
            layer=FindingLayer.TEETH,
            confidence=0.9,
            box_xyxy=None,
            polygon=[
                {"x": 10.0, "y": 10.0},
                {"x": 20.0, "y": 10.0},
                {"x": 15.0, "y": 20.0},
            ],
            accepted=True,
            source=FindingSource.AI,
            created_at=ts,
            updated_at=ts,
        )
        workspace.extend_findings(record, [gf])
        record.status = AnalysisStatus.REVIEWING
        record.alert_level = AlertLevel.LOW
        record.completed_at = ts

    monkeypatch.setattr(
        "backend.app.api.routes.analyses.process_guest_inference",
        fake_guest_inference,
    )

    with TestClient(app) as tc:
        yield tc

    app.dependency_overrides.clear()

    async def _dispose() -> None:
        await engine.dispose()

    asyncio.run(_dispose())
