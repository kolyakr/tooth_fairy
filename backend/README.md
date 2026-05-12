# ToothFairy FastAPI Backend

Modular async API (FastAPI + SQLAlchemy 2.0 + PostgreSQL) for panoramic X-ray uploads, YOLO inference (via `modeling/utils/pipeline.py`), structured findings, and HITL audit trails.

## Prerequisites

- **Python:** 3.10+ recommended (3.12 per repo `AGENTS.md`).
- **PostgreSQL:** local Docker Compose file included.
- **ML weights:** optional until you run live inference; install `requirements-ml.txt` and ensure `.pt` files exist under `modeling/models/`.

## Setup

```bash
conda activate ml-env   # or your venv
cd /path/to/tooth_fairy

# Database
docker compose -f backend/docker-compose.yml up -d

# Python deps (API only — tests do not require Torch)
pip install -r backend/requirements.txt

# Optional: Ultralytics / OpenCV / Torch for real inference
pip install -r backend/requirements-ml.txt

# Migrations (sync URL uses psycopg2 — matches TOOTHFAIRY_DATABASE_URL with +asyncpg swapped internally by Alembic env)
export TOOTHFAIRY_DATABASE_URL=postgresql+asyncpg://toothfairy:toothfairy@localhost:5432/toothfairy
alembic -c backend/alembic.ini upgrade head
```

Copy [`backend/.env.example`](.env.example) to the repo root as `.env` if you use pydantic-settings file loading.

## Run the server

From the **repository root** (so `backend` is importable as a package):

```bash
uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
```

- Interactive docs: `http://localhost:8000/docs`
- Health (no version prefix): `GET http://localhost:8000/health`
- Versioned API: `http://localhost:8000/api/v1/...`

## Layout

| Path | Role |
|------|------|
| `backend/app/main.py` | FastAPI app, CORS, lifespan (`model_registry`), `/health` + `/api/v1` |
| `backend/app/core/` | Settings, async engine, logging, `build_model_registry()` |
| `backend/app/db/` | SQLAlchemy models |
| `backend/app/schemas/` | Pydantic v2 contracts |
| `backend/app/services/` | Business logic + `process_analysis_inference` (background job) |
| `backend/app/api/routes/` | Routers (`patients`, `analyses`, `findings`, `health`) |
| `backend/alembic/` | Migrations |

## API cheat sheet (`curl`)

Replace `HOST=http://localhost:8000` and generated UUIDs as needed.

### Health

```bash
curl -s "$HOST/health"
```

### Create patient (optional — upload also upserts by `patient_code`)

```bash
curl -s -X POST "$HOST/api/v1/patients" \
  -H "Content-Type: application/json" \
  -d '{"patient_code":"P-4021","name":"Demo","age":45}'
```

### Upload OPG + run inference (async background task)

```bash
curl -s -X POST "$HOST/api/v1/analyses" \
  -F patient_code=P-4021 \
  -F patient_name="Demo Patient" \
  -F age=45 \
  -F scan_date=2026-05-02 \
  -F chief_complaint=pain \
  -F file=@/path/to/opg.png
```

Returns `{ "id": "<uuid>", "status": "Pending AI" }`. Poll until `status` is `Reviewing` or `Failed`.

### List analyses (dashboard)

```bash
curl -s "$HOST/api/v1/analyses?limit=50"
```

### Analysis detail (polling)

```bash
curl -s "$HOST/api/v1/analyses/<ANALYSIS_ID>"
```

### Fetch stored image (BYTEA)

```bash
curl -s -o original.jpg "$HOST/api/v1/analyses/<ANALYSIS_ID>/image?kind=original"
```

`kind`: `original`, `quadrants_overlay`, `quadrants_grid`, `teeth_overlay`, `periapical_full_overlay`, `periapical_quadrants_overlay`, `teeth_classification_overlay`.

### Findings

```bash
curl -s "$HOST/api/v1/analyses/<ANALYSIS_ID>/findings"
```

Create manual finding:

```bash
curl -s -X POST "$HOST/api/v1/analyses/<ANALYSIS_ID>/findings" \
  -H "Content-Type: application/json" \
  -H "X-Reviewer: Dr. Smith" \
  -d '{"tooth_label":"FDI-36","finding":"Manual Finding","confidence":1,"accepted":true,"polygon":[{"x":10,"y":10},{"x":20,"y":10},{"x":15,"y":20}],"layer":"periapical","source":"manual"}'
```

Update finding:

```bash
curl -s -X PATCH "$HOST/api/v1/findings/<FINDING_ID>" \
  -H "Content-Type: application/json" \
  -H "X-Reviewer: Dr. Smith" \
  -d '{"accepted":false}'
```

Delete finding:

```bash
curl -s -X DELETE "$HOST/api/v1/findings/<FINDING_ID>" -H "X-Reviewer: Dr. Smith" -w "%{http_code}"
```

### Audit

```bash
curl -s "$HOST/api/v1/analyses/<ANALYSIS_ID>/audit"
```

Append client audit row:

```bash
curl -s -X POST "$HOST/api/v1/analyses/<ANALYSIS_ID>/audit" \
  -H "Content-Type: application/json" \
  -d '{"reviewer":"Dr. Smith","action":"Opened viewer","action_type":"system"}'
```

Export audit JSON:

```bash
curl -s -o audit.json "$HOST/api/v1/analyses/<ANALYSIS_ID>/audit/export"
```

### Complete review (HITL gate)

```bash
curl -s -X POST "$HOST/api/v1/analyses/<ANALYSIS_ID>/complete-review" \
  -H "Content-Type: application/json" \
  -d '{"reviewer":"Dr. Smith"}'
```

## Production

See **[docs/DEPLOYMENT.md](../docs/DEPLOYMENT.md)** in the repository for environment variables, Docker (`Dockerfile.backend`, `docker-compose.prod.yml`), TLS, CORS, JWT and guest cookies, and database migrations.

## Tests

```bash
cd backend
pytest app/tests -q
```

Uses in-memory SQLite, mocks Ultralytics, and exercises upload → synthetic inference hook → review completion.

## Notes

- **Inference thread:** blocking `run_pipeline` runs inside `asyncio.to_thread` so the event loop stays responsive.
- **Shared weights:** `run_pipeline(..., model_registry=...)` accepts a shared `ModelRegistry` (see `modeling/utils/pipeline.py`).
- **BLOB storage:** raster artifacts live in `image_assets.data` (BYTEA); replace `ImageAsset` persistence in services if you later move to object storage.
