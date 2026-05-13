# ToothFairy

**Human-in-the-loop assistant for dental panoramic X-rays (OPG).** ToothFairy runs a staged computer-vision pipeline (quadrants, teeth, pathologies), then hands every finding to a clinician in a web viewer: toggle overlays, adjust confidence, accept or reject, edit geometry, and export an auditable, template-based PDF report. The system is built so **AI suggests; the dentist decides**—nothing is presented as a final diagnosis without review.

> **Disclaimer — models can be wrong.** Every inference stage is **probabilistic**: it can **miss** pathology, **flag** artefacts or normal anatomy, or **mislabel** a finding. Treat all model output as **assistive only**. It is **not** a substitute for clinical examination, corroborating imaging, or the judgment of a licensed dentist or radiologist.

---

## What it does

1. **Ingest** — Upload an OPG (e.g. PNG/JPEG) with patient metadata from the web app.
2. **Analyze** — The API runs inference (quadrant segmentation, tooth-related outputs, periapical and pathology-oriented heads where configured), persists artifacts in PostgreSQL, and tracks status from pending through review.
3. **Review** — Open the interactive viewer: pan/zoom on the radiograph, filter by confidence, show or hide layers (quadrants, teeth, periapical, caries, impacted), accept/reject findings, draw or adjust annotations, and record who did what.
4. **Report** — After review, draft clinical text, preview structured sections, then generate a **ReportLab** PDF (practice-style letterhead) with optional embedded figures and pathology crops (template-driven, not a generative “black box” unless you add that layer yourself).

---

## Why it exists

- **Audit trail** — Actions can be logged for compliance-minded workflows (accept/reject, edits, exports).
- **Coordinates** — Local crop predictions are mapped back to full-image space so the UI and PDFs stay consistent with the OPG.
- **FDI-aware workflow** — Findings are organized around tooth labels suitable for clinical habit (World Dental Federation numbering where the pipeline provides it).

---

## Architecture (diagram)

End-to-end flow from **input X-ray** through **quadrant crops**, **parallel heads per crop**, **global mapping**, clinician review, and **PDF**. GitHub renders the Mermaid block; the PNG is an optional static slide asset.

![ToothFairy pipeline overview (static PNG)](docs/readme-assets/pipeline-overview.png)

---

## Demo

<video controls playsinline width="720">
  <source src="https://raw.githubusercontent.com/kolyakr/tooth_fairy/main/uploads/tooth_fairy_test.mp4" type="video/mp4" />
</video>

[`uploads/tooth_fairy_test.mp4`](uploads/tooth_fairy_test.mp4)

---

## Model output examples (visual tests)

Examples below are committed under [`modeling/outputs/`](modeling/outputs/) for validation frame **train_67**, produced with **[`modeling/run_inference.py`](modeling/run_inference.py)** once local weights and image paths are configured. The training/validation tree under `data/` is **not** shipped in this repository (see [Data](#data)); use your own OPG inputs or a compatible layout when you re-run.

### Input (original)

![Input OPG train_67](modeling/outputs/train_67_original.jpg)

### Quadrant segmentation

![Quadrant overlay train_67](modeling/outputs/train_67_quadrants_overlay.jpg)

### Teeth segmentation

![Teeth overlay train_67](modeling/outputs/train_67_teeth_overlay.jpg)

### Periapical detector (on quadrant crops, composited)

![Periapical overlay train_67](modeling/outputs/train_67_periapical_quadrants_overlay.jpg)

### Teeth classification (caries / impacted / etc.)

![Teeth classification overlay train_67](modeling/outputs/train_67_teeth_classification_overlay.jpg)

---

## Key functionality

| Area | What you get |
|------|----------------|
| **Dashboard** | Landing experience with navigation toward uploads and recent work ([`frontend/src/app/page.tsx`](frontend/src/app/page.tsx)). |
| **Upload** | Drag-and-drop friendly flow, patient fields, and analyze action ([`frontend/src/app/upload/page.tsx`](frontend/src/app/upload/page.tsx)). |
| **Viewer** | High-resolution canvas, **confidence threshold** slider, **layer toggles** (quadrants, teeth, periapical, caries, impacted), findings list, polygon/bbox editing, undo-friendly mutations ([`frontend/src/components/viewer/`](frontend/src/components/viewer/)). |
| **Review gate** | Mark review complete before report generation (status-driven UI). |
| **Report workspace** | Clinical summary, impression, recommendations, reviewer confirmation → preview → PDF generate and download ([`frontend/src/components/viewer/props-panel.tsx`](frontend/src/components/viewer/props-panel.tsx)). |
| **Auth (optional)** | Sign-in / sign-up routes exist ([`frontend/src/app/sign-in/page.tsx`](frontend/src/app/sign-in/page.tsx)); guest cookie flow remains supported for unauthenticated uploads when the API allows it. |
| **API** | REST under `/api/v1` with OpenAPI docs when the server is running ([`backend/app/main.py`](backend/app/main.py)). |

**Inference code path** — Orchestration and remapping live under [`modeling/utils/pipeline.py`](modeling/utils/pipeline.py) (and related services); weights are **not** stored in this Git repository—see [Model weights](#model-weights-hugging-face) below.

---

## Data

Training and experimentation for this project combine **public dental radiography** with a **large merged corpus** I assembled and curated myself, plus additional images and labels I produced for this work. Part of the **quadrant segmentation** training data includes **self-annotated** OPG crops and masks.

The full consolidated dataset is **not** redistributed inside this repo (size, licensing mix, and ongoing cleanup). If you are a researcher or partner and want to **discuss access**, data use terms, or collaboration, **contact me** (e.g. via GitHub Issues or the contact options on my profile).

**Why caries / impacted training is separate from periapical** — Useful dental X-ray data is **mostly private**, so I rarely had one large, consistently labeled OPG corpus for every pathology at once. I **trained the teeth-classification head (caries, impacted teeth, etc.) on its own data and annotation pass**, and the **periapical detector on cropped quadrants** on a **different** setup. They are **two separate YOLO runs** in the pipeline, not a single end-to-end multi-label model trained on one shared private dataset.

---

## Model metrics (validation)

Figures below are taken from the **last training epoch** logged in each Ultralytics `results.csv` (validation metrics as reported during that run). **B** = bbox branch, **M** = mask branch (segmentation models log both; I quote **mask** for quadrant / teeth segmentation / teeth classification, and **bbox** for the periapical detector, which is detection-oriented on crops).

| Model | Source file | Epochs | P | R | mAP50 | mAP50-95 | Branch |
|-------|-------------|--------|---|---|-------|----------|--------|
| **Quadrant segmentation** | [`modeling/runs/quadrant segmentation/results.csv`](modeling/runs/quadrant%20segmentation/results.csv) | 39 | 99.49% | 99.24% | 99.48% | 71.17% | M |
| **Teeth segmentation** | [`modeling/runs/teeth segmentation/results.csv`](modeling/runs/teeth%20segmentation/results.csv) | 50 | 92.95% | 92.40% | 96.31% | 60.08% | M |
| **Teeth classification** (caries / impacted / …) | [`modeling/runs/teeth_classification/results.csv`](modeling/runs/teeth_classification/results.csv) | 45 | 73.46% | 75.36% | 78.28% | 51.33% | M |
| **Periapical (cropped)** | [`modeling/runs/periapical detector (crops)/periapical_cropped 1/results.csv`](modeling/runs/periapical%20detector%20%28crops%29/periapical_cropped%201/results.csv) | 50 | 69.14% | 62.65% | 65.48% | 29.13% | B |

> **Note:** mAP50 = mAP@IoU 0.5; mAP50-95 = COCO-style mean AP over IoU 0.5–0.95. Different runs use different splits and augmentations—treat these as **historical training snapshots**, not a formal benchmark leaderboard.

---

## Model weights (Hugging Face)

YOLO `.pt` checkpoints are **hosted separately** so the code repository stays lightweight:

**[ra1nbowdash/tooth-fairy-weights](https://huggingface.co/ra1nbowdash/tooth-fairy-weights/tree/main)** (Hugging Face — `models/` and related files).

### Where to put files locally

The API resolves weights under [`modeling/models/`](modeling/) by default (see [`backend/app/core/config.py`](backend/app/core/config.py)). After download, you should have **four** `best.pt` files at these paths (create folders with these exact names if needed):

| Model role | Path relative to repo root |
|------------|----------------------------|
| Quadrant segmentation | `modeling/models/quadrant segmentation/best.pt` |
| Teeth segmentation | `modeling/models/teeth segmentation/best.pt` |
| Periapical (cropped) | `modeling/models/periapical detector (cropped)/best.pt` |
| Teeth classification | `modeling/models/teeth classification/best.pt` |

If the Hugging Face repo layout matches a `models/` tree, you can often **merge that folder into** `modeling/models/` so the subfolders and filenames line up with the table above.

### How to download

**Option A — Browser:** open the [Files](https://huggingface.co/ra1nbowdash/tooth-fairy-weights/tree/main) view, download the `models` (or individual `.pt`) assets, then place them as in the table.

**Option B — Hugging Face CLI** (requires [the Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli) or `pip install huggingface_hub`):

```bash
# From repository root — downloads the whole repo snapshot into ./hf-tooth-fairy-weights
pip install huggingface_hub
huggingface-cli download ra1nbowdash/tooth-fairy-weights --local-dir ./hf-tooth-fairy-weights
```

Then copy or merge the downloaded `models` directory into `modeling/models/` so the four paths above exist.

**Option C — Direct URLs:** you can point [`scripts/fetch_ml_weights.py`](scripts/fetch_ml_weights.py) at raw HTTPS URLs using `TOOTHFAIRY_FETCH_WEIGHT_QUADRANTS_URL`, `TOOTHFAIRY_FETCH_WEIGHT_TEETH_URL`, `TOOTHFAIRY_FETCH_WEIGHT_PERIAPICAL_URL`, and `TOOTHFAIRY_FETCH_WEIGHT_TEETH_CLASSIFICATION_URL` (one per file), then run `python scripts/fetch_ml_weights.py` from the repo root.

You can override paths with `TOOTHFAIRY_MODEL_QUADRANTS_PATH`, `TOOTHFAIRY_MODEL_TEETH_PATH`, `TOOTHFAIRY_MODEL_PERIAPICAL_PATH`, and `TOOTHFAIRY_MODEL_TEETH_CLASSIFICATION_PATH` in `.env` if you keep weights elsewhere.

---

## Tech stack

| Layer | Choices |
|-------|---------|
| **API** | Python 3.12+, FastAPI, Pydantic v2, SQLAlchemy 2.0 (async), Alembic |
| **DB** | PostgreSQL (local dev via Docker Compose in `backend/`) |
| **Web** | Next.js (App Router), TypeScript, Tailwind CSS |
| **CV / ML (optional)** | PyTorch, Ultralytics, OpenCV — [`backend/requirements-ml.txt`](backend/requirements-ml.txt) |
| **PDF** | ReportLab templates — [`backend/app/services/report_pdf_templates.py`](backend/app/services/report_pdf_templates.py) |

---

## Prerequisites

- **Python** 3.12+ (conda or `venv` is fine)
- **Node.js** + npm (for the frontend)
- **Docker** (optional but recommended) — only to run PostgreSQL via [`backend/docker-compose.yml`](backend/docker-compose.yml)
- **Weights** — Not in Git; download from [Hugging Face](https://huggingface.co/ra1nbowdash/tooth-fairy-weights/tree/main) and install under `modeling/models/` as described in [Model weights](#model-weights-hugging-face)

---

## Local setup (new contributor)

### 1. Clone and environment files

```bash
git clone https://github.com/kolyakr/tooth_fairy.git
cd tooth_fairy
```

- **Backend:** copy [`backend/.env.example`](backend/.env.example) to a **repository root** `.env` (Pydantic loads `TOOTHFAIRY_*` from cwd when you run uvicorn from the root). Adjust `TOOTHFAIRY_DATABASE_URL` and `TOOTHFAIRY_CORS_ORIGINS` if needed.
- **Frontend:** copy [`frontend/.env.example`](frontend/.env.example) to `frontend/.env.local`. Default API base is `http://localhost:8000`.

### 2. Database

```bash
docker compose -f backend/docker-compose.yml up -d
```

Creates user/db `toothfairy` / `toothfairy` on `localhost:5432` (see compose file for overrides).

### 3. Backend

From the **repository root** (so `backend` is importable as a package):

```bash
conda activate ml-env   # or: python -m venv .venv && source .venv/bin/activate
pip install -r backend/requirements.txt
pip install -r backend/requirements-ml.txt   # optional — large; only if you run live YOLO inference

export TOOTHFAIRY_DATABASE_URL=postgresql+asyncpg://toothfairy:toothfairy@localhost:5432/toothfairy
alembic -c backend/alembic.ini upgrade head

uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
```

- Interactive docs: `http://localhost:8000/docs`
- Health: `http://localhost:8000/health`

### 4. Frontend

```bash
cd frontend
npm install
npm run dev
```

App: `http://localhost:3000`

### 5. Optional quality commands

```bash
# Backend tests (from backend/ — see backend/pytest.ini)
cd backend && pytest -q

# Frontend
cd frontend && npm run lint && npm run build
```

More `curl` examples: [`backend/README.md`](backend/README.md).

---

## Repository layout

| Path | Role |
|------|------|
| [`backend/`](backend/) | FastAPI app, schemas, services, Alembic migrations |
| [`frontend/`](frontend/) | Next.js UI: dashboard, upload, viewer, auth pages |
| [`modeling/`](modeling/) | Weights (`modeling/models/`), `utils/pipeline.py`, [`run_inference.py`](modeling/run_inference.py), sample visual outputs in [`modeling/outputs/`](modeling/outputs/) |
| [`uploads/`](uploads/) | Demo screencast: [`tooth_fairy_test.mp4`](uploads/tooth_fairy_test.mp4) |

---

## Privacy, PHI, and git hygiene

- Treat **patient imagery and identifiers as sensitive**. Do not commit PHI, real clinic data, or production secrets.
- Repo `.gitignore` excludes `.env` and common weight extensions (e.g. `*.pt`); keep secrets local and fetch weights from [Hugging Face](https://huggingface.co/ra1nbowdash/tooth-fairy-weights/tree/main) as documented above.
- Large binaries (long demo videos, checkpoints) are usually kept **out of Git** or via **Git LFS** / external storage—see the [Data](#data) section for the training corpus policy.

---

## Acknowledgements

Built with **FastAPI**, **Next.js**, **Ultralytics YOLO**, **PyTorch**, **ReportLab**, and related open-source libraries listed in `backend/requirements.txt` and `frontend/package.json`.

---

## License

This project is licensed under the **MIT License** — see [`LICENSE`](LICENSE).

Copyright © 2026 @kolyakr.
