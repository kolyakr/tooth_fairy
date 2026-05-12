# ToothFairy

Multi-stage AI assistance for dental panoramic X-rays (OPG): quadrant segmentation, tooth masks, periapical detection, and a **Next.js** viewer for dentist review (human-in-the-loop). A **FastAPI** backend persists analyses, findings, audit trails, and PDF reports.

## Repository layout

| Path | Role |
|------|------|
| [`backend/`](backend/) | FastAPI API, SQLAlchemy models, Alembic migrations, inference orchestration |
| [`frontend/`](frontend/) | Next.js 16+ app (dashboard, upload, interactive viewer) |
| [`modeling/`](modeling/) | YOLO weights (`modeling/models/`), research notebooks, `utils/pipeline.py` used at inference time |
| [`AGENTS.md`](AGENTS.md) | Contributor rules, stack defaults, and product notes |

## Quick start (local development)

**Backend** (from repository root so `backend` is importable as a package):

```bash
conda activate ml-env   # or another Python 3.12+ environment
pip install -r backend/requirements.txt
# Optional: live inference (Torch, Ultralytics, OpenCV)
pip install -r backend/requirements-ml.txt

docker compose -f backend/docker-compose.yml up -d
export TOOTHFAIRY_DATABASE_URL=postgresql+asyncpg://toothfairy:toothfairy@localhost:5432/toothfairy
alembic -c backend/alembic.ini upgrade head

uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
```

**Frontend:**

```bash
cd frontend
cp .env.example .env.local   # set NEXT_PUBLIC_API_BASE_URL if the API is not localhost:8000
npm install
npm run dev
```

- API docs: `http://localhost:8000/docs`
- Health: `GET http://localhost:8000/health`
- App: `http://localhost:3000`

More API examples: [`backend/README.md`](backend/README.md).

## Deployment

See **[docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)** for production environment variables, Docker Compose, CORS, auth/guest cookies, database migrations, ML weights, and split-hosting notes.

**Docker (quick reference):** `docker compose -f docker-compose.prod.yml up --build` uses [deploy/compose.env](deploy/compose.env) for local smoke defaults. For production, use [`.env.production.example`](.env.production.example) and wire secrets per [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md). Use profile `with-frontend` to include the Next.js container (`docker compose -f docker-compose.prod.yml --profile with-frontend up --build`).

## License / data

Treat patient imagery and metadata as sensitive. Do not commit real PHI or production `.env` files.
