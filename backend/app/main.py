"""FastAPI application entrypoint."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import APIRouter, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from sqlalchemy import text

from backend.app.api.routes import analyses, auth, findings, health, patients
from backend.app.core.config import get_settings
from backend.app.core.database import engine
from backend.app.core.logging import configure_logging, get_logger
from backend.app.core.model_registry import build_model_registry
from backend.app.services.domain_errors import DomainError
from backend.app.services.guest_workspace import GuestWorkspace

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Configure shared resources for request handlers."""
    configure_logging()
    log = get_logger(__name__)
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
    except Exception as exc:
        log.error(
            "PostgreSQL is not reachable (%s: %s). Endpoints that use the database will fail until it is "
            "running. From the repository root start the bundled Postgres:\n"
            "  docker compose -f backend/docker-compose.yml up -d\n"
            "Then apply migrations once:\n"
            "  alembic -c backend/alembic.ini upgrade head",
            type(exc).__name__,
            exc,
        )
    app.state.model_registry = build_model_registry()
    app.state.guest_workspace = GuestWorkspace(settings)
    yield
    await engine.dispose()


app = FastAPI(title=settings.project_name, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(DomainError)
async def domain_error_handler(_, exc: DomainError) -> JSONResponse:
    """Map domain errors to JSON responses."""
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.message})


app.include_router(health.router)

api_router = APIRouter(prefix=settings.api_v1_prefix)
api_router.include_router(auth.router)
api_router.include_router(patients.router)
api_router.include_router(analyses.router)
api_router.include_router(findings.router)
app.include_router(api_router)
