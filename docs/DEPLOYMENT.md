# ToothFairy deployment guide

This document describes how to run ToothFairy in production: architecture, environment variables, containers, database migrations, ML artifacts, and security checklists.

## Architecture

```mermaid
flowchart LR
  subgraph clients [Clients]
    Browser[Browser]
  end
  subgraph edge [Application tier]
    Next[Next.js]
    API[FastAPI]
  end
  subgraph data [Data tier]
    PG[(PostgreSQL)]
    Weights[YOLO weights]
  end
  Browser --> Next
  Browser -->|"fetch Bearer JWT and cookies"| API
  Next -->|"NEXT_PUBLIC_API_BASE_URL"| API
  API --> PG
  API --> Weights
```

- **Next.js** serves the UI. The browser calls the **FastAPI** API directly using `NEXT_PUBLIC_API_BASE_URL` (set at **build time**).
- **PostgreSQL** stores patients, analyses, BYTEA image assets, findings, audit rows, and generated PDF report blobs.
- **Guest mode** uses an in-memory workspace keyed by a signed **HTTP-only cookie** (no DB rows for that upload path). **Persisted mode** uses **JWT** (`Authorization: Bearer`) for API routes that distinguish principal vs guest.
- **Important:** Plain `<img src="...">` and `window.open(apiUrl)` do **not** send `Authorization`. The frontend uses `fetch` with Bearer + `credentials: "include"` for images, report PDF download, and audit export (see `frontend/src/lib/api-client.ts`).

## Environment variables (backend)

All backend settings use the prefix `TOOTHFAIRY_` and are defined in [`backend/app/core/config.py`](../backend/app/core/config.py). Copy [`backend/.env.example`](../backend/.env.example) to the host or Compose `env_file` (typically repo root `.env` when running uvicorn from the repo root).

| Variable | Type | Default / notes | Production |
|----------|------|-----------------|------------|
| `TOOTHFAIRY_DATABASE_URL` | string | `postgresql+asyncpg://toothfairy:toothfairy@localhost:5432/toothfairy` | **Required:** managed Postgres URL (asyncpg driver). |
| `TOOTHFAIRY_CORS_ORIGINS` | JSON list | `["http://localhost:3000"]` | **Required:** every browser origin that calls the API (e.g. `["https://app.example.com"]`). Use JSON array syntax as in `.env.example`. |
| `TOOTHFAIRY_AUTH_JWT_SECRET` | string or unset | unset | **Strongly recommended:** set a long random secret. Used to verify Bearer JWTs **and** to sign guest session cookies. If unset, guest signing uses a **development-only** fallback (see `guest_cookie.py`). |
| `TOOTHFAIRY_AUTH_JWT_ALGORITHM` | string | `HS256` | Usually leave default. |
| `TOOTHFAIRY_AUTH_ACCESS_TOKEN_TTL_MINUTES` | int | `60` (5–1440) | Access token lifetime for issued JWTs. |
| `TOOTHFAIRY_AUTH_DEV_LOGIN_ENABLED` | bool | `false` | **Must be `false` in production.** When `true`, exposes `POST /api/v1/auth/token` for arbitrary user-id JWT minting. |
| `TOOTHFAIRY_GUEST_SESSION_COOKIE_NAME` | string | `toothfairy_guest` | Cookie name for guest workspace. |
| `TOOTHFAIRY_GUEST_SESSION_TTL_SECONDS` | int | `86400` | Guest cookie max-age. |
| `TOOTHFAIRY_GUEST_MAX_ANALYSES_PER_SESSION` | int | `50` | Cap guest analyses per session id. |
| `TOOTHFAIRY_GUEST_COOKIE_SECURE` | bool | `false` | Set **`true` behind HTTPS** so the guest cookie is not sent over plain HTTP. |
| `TOOTHFAIRY_GUEST_COOKIE_SAMESITE` | string | `lax` | One of `lax`, `strict`, `none`. Use `none` only if the API is on a **different site** than the Next app and you use HTTPS + `Secure`; otherwise prefer `lax`. |
| `TOOTHFAIRY_DEBUG` | bool | `false` | Keep `false` in production. |
| `TOOTHFAIRY_PROJECT_NAME` | string | API title | Optional branding. |
| `TOOTHFAIRY_API_V1_PREFIX` | string | `/api/v1` | Rarely changed. |
| `TOOTHFAIRY_REPO_ROOT` | path | auto from package location | Override if the repo is mounted unexpectedly in containers. |
| `TOOTHFAIRY_MODELING_ROOT` | path | `<repo>/modeling` | Root containing `utils/` and `models/`. |
| `TOOTHFAIRY_MODEL_QUADRANTS_PATH` | path | under `modeling/models/...` | Override per-stage `.pt` path. |
| `TOOTHFAIRY_MODEL_TEETH_PATH` | path | … | |
| `TOOTHFAIRY_MODEL_PERIAPICAL_PATH` | path | … | |
| `TOOTHFAIRY_MODEL_TEETH_CLASSIFICATION_PATH` | path | … | |
| `TOOTHFAIRY_CONF_QUADRANTS` | float | `0.3` | Inference confidence thresholds. |
| `TOOTHFAIRY_CONF_TEETH` | float | `0.3` | |
| `TOOTHFAIRY_CONF_PERIAPICAL` | float | `0.3` | |
| `TOOTHFAIRY_CONF_TEETH_CLASSIFICATION` | float | `0.3` | |

## Environment variables (frontend)

| Variable | When | Notes |
|----------|------|--------|
| `NEXT_PUBLIC_API_BASE_URL` | **Build time** | Public base URL of the API **without** trailing slash, e.g. `https://api.example.com`. Next.js inlines this into the client bundle. |

See [`frontend/.env.example`](../frontend/.env.example).

## Database migrations

Run from the **repository root** (Alembic is configured in `backend/alembic.ini`):

```bash
export TOOTHFAIRY_DATABASE_URL=postgresql+asyncpg://USER:PASS@HOST:5432/DBNAME
alembic -c backend/alembic.ini upgrade head
```

Use a managed Postgres backup/restore strategy appropriate to your provider (snapshots, `pg_dump`, point-in-time recovery). Application BLOBs live in Postgres today; restoring the DB restores images and PDFs.

## ML weights and inference dependencies

- Default weight paths are under `modeling/models/` (see `resolved_model_paths()` in config). **Do not** commit large `.pt` binaries to git unless you explicitly want them in the image; prefer **volume mounts** or a private artifact fetch in your CI/CD pipeline.
- `backend/requirements.txt` is enough for API + tests without Torch.
- Install `backend/requirements-ml.txt` for **Ultralytics / Torch / OpenCV** inference (large image, CPU or GPU base image as needed).

## Docker (production-oriented)

The repo provides:

- [`Dockerfile.backend`](../Dockerfile.backend) — API image. Build from the **repository root**. Optional build-arg `INSTALL_ML=true` installs `backend/requirements-ml.txt` (large). YOLO `.pt` files are excluded from the build context via [`.dockerignore`](../.dockerignore); mount weights at runtime (see Compose below).
- [`Dockerfile.frontend`](../Dockerfile.frontend) — Next.js **standalone** production image (`node server.js`). Build-arg `NEXT_PUBLIC_API_BASE_URL` must match the URL browsers use to reach the API.
- [`docker-compose.prod.yml`](../docker-compose.prod.yml) — Postgres + `api`. Optional `web` service: `docker compose -f docker-compose.prod.yml --profile with-frontend up --build`.

First-time Compose (uses tracked [deploy/compose.env](../deploy/compose.env) for local smoke tests):

```bash
docker compose -f docker-compose.prod.yml up --build
```

For production secrets, copy [`.env.production.example`](../.env.production.example) to a **gitignored** `.env.production`, edit values, then point the `api` service `env_file` at that file (or inject variables via your orchestrator).

Apply migrations using the API container (inherits `deploy/compose.env`; DB host is `db` on the Compose network):

```bash
docker compose -f docker-compose.prod.yml run --rm api alembic -c backend/alembic.ini upgrade head
```

Alternatively, from the host with `TOOTHFAIRY_DATABASE_URL` pointing at Postgres:

```bash
export TOOTHFAIRY_DATABASE_URL=postgresql+asyncpg://toothfairy:YOUR_PASSWORD@localhost:5432/toothfairy
alembic -c backend/alembic.ini upgrade head
```

API-only Compose binds `./modeling/models` read-only into the container (`MODELING_MODELS_HOST_PATH` overrides the host path). Ensure weight files exist there when `INSTALL_ML=true`, or ship a custom image that copies weights.

Do **not** commit real secrets. [deploy/compose.env](../deploy/compose.env) contains **smoke-test defaults only**; for production use a gitignored `.env.production` (see [`.env.production.example`](../.env.production.example)) or your platform's secret store.

## Operations

- **Liveness / readiness:** `GET /health` (no `/api/v1` prefix).
- **Logs:** structured logging is configured in `backend/app/core/logging.py`; follow your platform’s log aggregation.
- **Resources:** inference is CPU-heavy (or GPU if you configure Torch accordingly). Background inference runs in a thread pool; size workers and DB pool for expected concurrency.
- **Guest workspace:** `GuestWorkspace` is **in-memory**. Scale the API to **multiple replicas** only with awareness: guest sessions are not shared across instances. Use **sticky sessions** or a single replica for guest-heavy workloads until guest state is externalized (out of scope for current code).

## Split hosting (example)

| Layer | Example | Notes |
|-------|---------|--------|
| Frontend | Vercel / Netlify / static host | Build with `NEXT_PUBLIC_API_BASE_URL` pointing at the real API hostname. |
| API | Fly.io, Render, ECS, VM + Docker | Set `TOOTHFAIRY_CORS_ORIGINS` to the deployed Next origin. HTTPS + `TOOTHFAIRY_GUEST_COOKIE_SECURE=true` for guest uploads from that origin. |
| Database | RDS, Cloud SQL, managed Postgres | Set `TOOTHFAIRY_DATABASE_URL`. |

If the UI and API are on **different registrable domains**, guest `SameSite=Lax` may not send the cookie on all cross-site subrequests; you may need `SameSite=None` + `Secure` and careful CORS testing.

## Security checklist (production)

1. Set **`TOOTHFAIRY_AUTH_JWT_SECRET`** to a strong random value (rotate periodically).
2. Set **`TOOTHFAIRY_AUTH_DEV_LOGIN_ENABLED=false`** (default).
3. Set **`TOOTHFAIRY_CORS_ORIGINS`** to an explicit allowlist (no wildcard when using `allow_credentials=True`).
4. Enable **`TOOTHFAIRY_GUEST_COOKIE_SECURE=true`** when serving the API over HTTPS.
5. If you set **`TOOTHFAIRY_GUEST_COOKIE_SAMESITE=none`** (cross-site API + SPA), browsers require **`TOOTHFAIRY_GUEST_COOKIE_SECURE=true`** as well.
6. Use TLS everywhere; terminate TLS at your reverse proxy or load balancer.
7. Replace dev JWT login with a real IdP (OAuth/OIDC) when you go beyond internal pilots (application change; not covered here).

## FAQ

**Docker Compose picked the wrong `TOOTHFAIRY_DATABASE_URL`.**  
Compose interpolates variables from a project-level `.env` file if present. Rename or adjust conflicting keys, or rely on `env_file: deploy/compose.env` for container env (see [docker-compose.prod.yml](../docker-compose.prod.yml)).

**Browser console: “The message port closed before a response was received.”**  
This usually comes from a **browser extension** (e.g. React DevTools, ad blockers), not from ToothFairy. Disable extensions or ignore if the Network tab shows successful API calls.

**401 / 404 on images or PDF download while “signed in”**  
Ensure the client uses **fetch** with the Bearer token (not raw `img src` or `window.open` to protected URLs). The shipped frontend already does this for viewer images and report/audit downloads.

## Follow-ups (not implemented in this repo)

- OAuth/OIDC instead of dev token issuance.
- Object storage (S3/GCS) for BYTEA assets and PDFs instead of Postgres.
- Shared guest session store (Redis) for multi-replica guest mode.
