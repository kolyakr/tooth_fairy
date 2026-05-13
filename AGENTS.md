# ToothFairy: Project context and contributor notes

This file summarizes architecture, conventions, and product expectations for **ToothFairy** — a multi-stage workflow for dental panoramic X-rays (OPGs): computer vision stages, a clinician-facing web app, and structured PDF export after human review.

## 🎯 Project Overview

A two-stage hierarchical pipeline:

1. **Stage 1 (Global):** Segment 4 quadrants of the OPG.
2. **Intermediate:** Crop original image into 4 quadrant-based images.
3. **Stage 2 (Local Parallel):** Process crops through three specialized models (Teeth Enumeration [FDI], Disease Segmentation, and Periapical Detection).
4. **Final:** Aggregate annotations back to the global view, provide a **Next.js** editor for doctors, and generate **template-based PDF reports** (clinician-edited narrative; ReportLab), after review.

---

## 📂 Folder structure and permissions


| Path         | Purpose                                                | Guideline                                                                                                        |
| ------------ | ------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------- |
| ------------ | ------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------- |
| `/data`      | Raw/Processed datasets, YOLO formats.                  | **READ-ONLY.** Never rewrite, move, or index for code suggestions.                                               |
| `/models`    | Weights and architecture definitions.                  | Contains specialized sub-folders for each task.                                                                  |
| `/notebooks` | Research, data merging, training logs.                 | **Reference, where you can look at coding history.** Read markdowns here to understand inference/cropping logic. |
| `/utils`     | Shared helper functions (image processing, FDI logic). | **Reference, where you can look at coding history.** Primary location for reusable logic.                        |
| `/backend`   | FastAPI application.                                   | Use strict folder separation (routes, schemas, services).                                                        |
| `/frontend`  | Next.js application.                                   | Interactive UI for bbox/mask editing (Roboflow-style).                                                           |


---

## 🚀 Quick Start Commands

Use these defaults unless the task calls for something different.

### Python / Backend

- `conda activate ml-env`
- `python -V` (expect Python 3.12.x)
- `pip install -r requirements.txt` (if dependencies are missing)
- `pytest -q` (run tests)
- `uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000` (run API from repo root)

### Frontend

- `cd frontend`
- `npm install` (if dependencies are missing)
- `npm run dev` (run Next.js app)
- `npm run lint` (run lint checks)
- `npm run build` (production build sanity check)

---

## 🛠 Technical Stack & Environment

- **Environment:** Conda (`conda activate ml-env`)
- **Hardware:** Optimized for **macOS (M1 Pro)**. Use `torch.device("mps")` for local inference.
- **Backend:** Python 3.12, FastAPI, Pydantic v2, SQLAlchemy/Motor (if needed).
- **Frontend:** Next.js (TypeScript), Tailwind CSS, Lucide-React.
- **ML Frameworks:** Ultralytics (YOLOv8/v11), PyTorch, opencv.

---

## 🧠 Architectural Guidelines

### 1. The Inference Pipeline

When writing inference code, follow this sequence:

1. Initialize `QuadrantSegmenter` (use this canonical name in new code and docs).
2. Pass OPG -> Get 4 crops.
3. **Parallel Execution:** Use `asyncio` or threading to run `TeethEnumerator`, `DiseaseSegmenter`, and `PeriapicalDetector` on the crops simultaneously.
4. Transform local crop coordinates back to global OPG coordinates.

### 2. Dental Specifics

- **FDI Numbering:** Teeth enumeration must follow the FDI World Dental Federation notation.
- **Diseases:** Target classes are `Caries`, `Deep Caries`, and `Impacted`.
- **Hierarchy:** Local detections (Stage 2) are parented to the Quadrant (Stage 1).

### 3. Notebook vs Production Code Policy

- Notebooks are research references and visual logic references.
- Reusable logic must be implemented in `/utils` or `/backend/services`, not copied inline from notebooks into routes.

### 4. Coordinate System Contract

- Use pixel coordinates unless explicitly documented otherwise.
- Bounding boxes use `[x_min, y_min, x_max, y_max]` in image coordinate space.
- Masks must include shape metadata compatible with global coordinate remapping.
- Every Stage 2 local prediction must be remapped to global OPG coordinates before API response.

---

## 📝 Coding Standards

- **Clean Code:** Follow SOLID principles. Keep business logic in `services/` and model loading in `core/`.
- **Documentation:** All functions must have Google-style docstrings in **English**.
- **Type Safety:** Strict use of Python type hints and TypeScript interfaces.
- **Visualization:** Refer to `/notebooks` or `/utils` for OpenCV/Matplotlib patterns from research; refine as needed.

---

## ✅ Change Strategy (Default)

When implementing a feature, prefer this order:

1. Update or define schemas/contracts.
2. Implement or update core service logic.
3. Connect route/controller layer.
4. Add/update tests.
5. Validate with quick manual run (sample inference or API call).

---

## 🧪 Testing Expectations

- Backend logic changes: add or update `pytest` coverage.
- Geometry/remapping changes: include at least one round-trip coordinate test (local -> global).
- Frontend annotation/editor changes: run lint and provide manual verification steps.
- Model integration changes: include a smoke test path (single image inference) when possible.

---

## 🔐 Safety and File Operation Rules

- `/data` is read-only: never modify, move, or delete files there.
- Do not modify model weight binaries (`.pt`, `.onnx`, `.engine`) unless explicitly requested.
- Treat generated notebook export artifacts as non-source files unless explicitly requested for commit.
- Avoid committing large generated binary artifacts by default.

---

## ❓ When to Ask Clarifying Questions

Ask before proceeding if any of these are unclear:

- Expected API response schema for a new or changed endpoint.
- Acceptance metric/threshold for ML behavior changes.
 
---

## 🖥 Frontend Product Requirements (Current Priority)

Model training is still in progress for some stages. Until all models are production-ready, prioritize frontend delivery with hard-coded/mock data and realistic interaction flows.

For medical applications, the UI must follow a strict Human-in-the-Loop (HITL) philosophy: AI is an assistant, and the dentist is the final decision-maker.

HITL enforcement rules:

- No auto-final diagnosis: the system must never present AI output as a final diagnosis without dentist confirmation.
- Explicit confirmation step: report generation/export must require a clear dentist review-and-confirm action.
- Full audit trail: persist accepted/rejected findings, manual edits (added/removed/reclassified polygons), reviewer identity, and timestamps.

### 1) Dashboard (Landing Page)

The dashboard is the daily command center for dentists.

- Keep layout clean and clinically usable.
- Follow HIPAA/GDPR-safe behavior (do not expose sensitive patient data without explicit authorization).
- Include top navigation with:
  - Search (Patient ID or Name)
  - User profile
  - Settings
- Main table: "Recent Analyses" with:
  - Patient name
  - Date of scan
  - Status (`Pending AI`, `Reviewed`, `Report Generated`)
  - AI alert level (visually emphasize severe findings)
- Include sorting/filtering.
- Include a prominent "New Patient Scan" action button.

### 2) Upload Workspace

This page must support a fast, one-screen flow.

- Provide a central drag-and-drop upload zone.
- Provide a sidebar or modal for patient metadata:
  - Patient ID
  - Age
  - Date of X-ray
  - Chief complaint
- File support target:
  - Required: `.jpg`, `.png`
  - Planned: `.dcm` (DICOM)
- "Analyze" action should support staged progress feedback:
  - `Segmenting Quadrants...`
  - `Detecting Pathologies...`

### 3) Interactive Viewer (Core)

This is the highest-priority and most technically complex UI.

- Left sidebar: grouped findings list by FDI tooth number.
- Center: high-resolution image canvas with smooth zoom/pan.
- Right sidebar or floating toolbar: annotation tools.

Required interactions:

- Toggle layers on/off (masks, boxes, labels).
- Toggle between:
  - Full panoramic view
  - 4 quadrant views
- Confidence threshold slider for dynamic filtering.
- Accept/Reject finding actions.
- Modify polygon vertices by drag.
- Add finding with draw tool and class assignment.
- Reclassify an existing finding.

Implementation guidance:

- Prefer `react-konva` or `fabric.js` for performant polygon editing and layered rendering.

### 4) Report Generator

After dentist review/approval:

- Show split view:
  - Rich text editor
  - Live PDF preview
- Auto-draft a clinical summary from confirmed findings.
- Auto-attach visual evidence crops with overlays.
- Provide treatment plan input field.
- Export options:
  - PDF
  - Print
  - Email

### Frontend Build Mode (Until Models Stabilize)

- Use hard-coded/mock payloads to implement UI now.
- Keep response contracts aligned with backend target schemas.
- Build components to be backend-pluggable:
  - Isolate API layer
  - Keep mock adapters replaceable
  - Avoid coupling UI logic to temporary mock structure
