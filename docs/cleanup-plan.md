# Cleanup Plan — UNSW-NB15_MLOPS

Canonical decisions live in [`docs/architecture-decisions.md`](./architecture-decisions.md).
This file is the ordered execution checklist. Work top to bottom within each pass.
Check off items as they are completed (`[x]`).

---

## Pass 1 — Identity Cleanup

**Goal:** Every place the project has a name, make it say the right thing.
Complete this pass before touching runtime files.

### 1.1 README title and header ✅

File: `README.md`

- [x] Change title from `# UNSW-NB15\_mlops` to `# UNSW-NB15_MLOPS`
- [x] Change architecture diagram label `FastAPI (app.py)` to `FastAPI (ids_unsw/serve/app.py)`
- [x] In Repo Layout section, remove root-level `app.py` and `app_dash.py` entries — they do not exist at root
- [x] Add `ids_unsw/serve/app.py` and `ids_unsw/ui/app_dash.py` to the layout listing
- [x] In Quick Start → Local → run command, change `uvicorn app:app` to `uvicorn ids_unsw.serve.app:app` (bash and PowerShell)
- [x] In Serving API section, change the `uvicorn app:app` command to `uvicorn ids_unsw.serve.app:app`
- [x] In Dash UI section, change `python app_dash.py` to `python ids_unsw/ui/app_dash.py`
- [x] Fix Troubleshooting `/health` entry — it is public, 401 means middleware misconfiguration
- [x] Add `IDS_EXPOSE_DOCS` to the Configuration & Environment Variables table
- [x] Update Docker Compose section to describe all three compose files with their purposes
- [x] Fix K8s probes description — remove auth requirement from `/health` probe docs

### 1.2 docker-compose.yml — service naming ✅

File: `docker-compose.yml`

- [x] Rename service `tensorflow` → `mlops-dev`
- [x] Change `container_name: UNSW-NB15_mlops` → `container_name: unsw-mlops-dev`
- [x] Change `image: tf-gpu-fixed` → `image: unsw-mlops-dev`
- [x] GPU env vars kept (GPU passthrough still needed); added comment explaining why

### 1.3 CI workflow naming ✅

File: `.github/workflows/ci.yml`

- [x] Change workflow `name: CI` → `name: UNSW-NB15_MLOPS CI`
- [x] Rename `data` job step to "Smoke: create synthetic raw dataset"
- [x] Rename `features` job step to "Smoke: build synthetic features"
- [x] Rename `train` job step to "Smoke: synthetic train — pipeline contract check (not real training)"
- [x] Added block comment above smoke fixture jobs explaining their purpose
- [x] `/health` health-check step no longer sends auth header — public endpoint
- [x] `env.IMAGE_NAME` updated to `${{ secrets.DOCKERHUB_USERNAME }}/unsw-nb15-mlops-api`

File: `.github/workflows/deploy.yml`

- [x] `/health` wait step no longer sends `Authorization` header — public endpoint
- [x] Workflow name updated to `UNSW-NB15_MLOPS — Build & Deploy to EKS`
- [x] `API_IMAGE` → `unsw-nb15-mlops-api`; `UI_IMAGE` → `unsw-nb15-mlops-ui`
- Note: `set image deploy/ids-api` and rollout targets are K8s Deployment names, not image slugs — unchanged and correct

### 1.4 Remove legacy TensorFlow identity residue ✅

- [x] `tf_gpu_project` — searched; no occurrences found in non-notebook files
- [x] `tf-gpu-fixed` — replaced with `unsw-mlops-dev` in `docker-compose.yml`
- [x] `tensorflow` service/container name — replaced with `mlops-dev` in `docker-compose.yml`
- [x] `Dockerfile.dev` `FROM tensorflow/...` kept (functional base image); added clarifying comment and changed `WORKDIR /tf` → `WORKDIR /workspace`

---

## Pass 2 — Truth Cleanup

**Goal:** README, metadata, auth rules, and env vars all tell the same story.

### 2.1 Metadata schema — production bundle ✅

Files:
- `notebooks/ids_unsw/models/bundle_xgb/metadata.json`
- `notebooks/ids_unsw/models/metadata.json`

- [x] Removed `"rf"` block, nested `"xgboost"` block, `"champion"` field, inline `"features"` array
- [x] Both files now use the canonical schema: `schema_version`, `project_name`, `model_family`, `artifact_format`, `n_features`, `threshold`, `feature_source`, `metrics_at_threshold` (all lowercase keys), `training_context`
- [x] Both files are identical for now; outer `models/metadata.json` kept as notebook workflow intermediate — non-canonical for serving

### 2.2 Serving app — auth contract ✅

File: `ids_unsw/serve/app.py`

- [x] `/health` is exempt from auth via path check in `require_token` — confirmed
- [x] `/health` returns `{"status": "ok", ...}` — confirmed public, minimal
- [x] All scoring/admin endpoints require Bearer token — confirmed
- [x] Added `IDS_EXPOSE_DOCS` env var: `false` by default; disables `/docs` and `/openapi.json` via FastAPI constructor
- [x] Added `@app.on_event("startup")` validation: checks all bundle files exist, `threshold` key present, warns if `schema_version` missing — fails fast with clear error

### 2.3 README — Docker Compose section ✅

- [x] Done as part of Pass 1.1 — Compose section now describes all three files with purpose labels

### 2.4 K8s docs — ConfigMap/Secret claims ✅

- [x] K8s probe description fixed — no longer implies `/health` requires auth (done in Pass 1.1)
- [ ] Confirm Secret/ConfigMap generators or manual creation instructions exist in manifests — deferred to Pass 3

### 2.5 Pass 2 verification sweep ✅

Verified after all Pass 2 edits:

- [x] `/health` auth: `require_token` exempts only `/health`; path check confirmed in `app.py` line 40
- [x] `IDS_EXPOSE_DOCS`: reads env var, defaults `false`; passes `docs_url`/`openapi_url` to FastAPI constructor
- [x] README commands: no remaining `uvicorn app:app` or `python app_dash.py` at root; `mlflow-docker-compose.yml` replaced with `docker-compose.mlflow.yml` throughout README (file rename deferred to Pass 3.1)
- [x] Metadata files: both contain `schema_version`, `project_name`; no `rf`, `xgboost` nested block, or `champion` keys
- [x] Scoring/admin auth: single global `dependencies=[Depends(require_token)]` on the FastAPI app — all endpoints covered
- [x] `Dockerfile.api` HEALTHCHECK: removed `Authorization: Bearer` header (public endpoint); also corrected env var names from `MODEL_BASE`/`BUNDLE_DIR` to canonical `IDS_BUNDLE_DIR`/`IDS_SCALER_PATH`

---

## Pass 3 — Runtime Cleanup

**Goal:** Files, directories, and configs reflect the current system accurately.

### 3.1 Rename compose files by purpose ✅

- [x] Renamed `docker-compose.yml` → `docker-compose.dev.yml`
- [x] Renamed `mlflow-docker-compose.yml` → `docker-compose.mlflow.yml`
- [x] Created `docker-compose.serve.yml` with `ids-api` and `dash-ui` services, canonical env vars, healthcheck, and `depends_on`
- [x] README already updated in Pass 1/2 to reference all three files correctly

### 3.2 Dockerfiles — env var alignment ✅

- [x] `Dockerfile.api`: uses `IDS_BUNDLE_DIR`, `IDS_SCALER_PATH`; HEALTHCHECK does not send auth; installs from `requirements/api.txt`
- [x] `Dockerfile.ui`: rewritten — uses `requirements/ui.txt`, copies only `ids_unsw/`, uses `gunicorn ids_unsw.ui.app_dash:server`
- [x] `ids_unsw/ui/Dockerfile.ui` (duplicate, referenced root `app_dash.py` that doesn't exist) — deleted; root `Dockerfile.ui` is canonical
- [x] `Dockerfile.dev`: TF identity comment added, `WORKDIR /tf` → `/workspace` (done in Pass 1)

### 3.3 Kubernetes — rename Streamlit → Dash UI ✅

- [x] `k8s/base/streamlit.yaml` deleted; `k8s/base/dash-ui.yaml` created with `dash-ui` Deployment and Service; image updated to `unsw-nb15-mlops-ui:latest`
- [x] `k8s/base/kustomization.yaml` updated to reference `dash-ui.yaml`
- [x] `k8s/base/ids-api.yaml` image updated from `ids-unsw-api:local` → `unsw-nb15-mlops-api:local`
- [x] `k8s/overlays/dev/kustomization.yaml`: patch target updated from `streamlit` → `dash-ui`; pre-existing `metrics-server-svc-patch` bug fixed (removed from overlay — belongs in `k8s/system/metrics-server/`)
- [x] `k8s/overlays/dev/probes-and-resources.yaml`: `name: streamlit` → `name: dash-ui`
- [x] `k8s/overlays/dev/streamlit-resources.yaml` deleted; `dash-ui-resources.yaml` created
- [x] `k8s/overlays/eks-dev/kustomization.yaml`: removed dead `$patch: delete` blocks for `streamlit`; `dash-ui.yaml` moved from `resources` to `patches` (strategic merge) to avoid duplicate with base
- [x] `k8s/overlays/eks-dev/dash-ui.yaml`: converted to strategic-merge patch; image updated to `unsw-nb15-mlops-ui:latest`
- [x] `k8s/overlays/eks-dev/probes-and-resources.yaml`: `name: streamlit` → `name: dash-ui`
- [x] `k8s/overlays/eks-dev/streamlit-resources.yaml` deleted; `dash-ui-resources.yaml` created
- [x] Verified: `kubectl kustomize k8s/base` ✓, `kubectl kustomize k8s/overlays/dev` ✓, `kubectl kustomize k8s/overlays/eks-dev` ✓
- [x] `k8s/archive/_ARCHIVE.md` added

### 3.4 Fix kustomization2.yaml ✅

- [x] `k8s/system/metrics-server/kustomization2.yaml` renamed to `kustomization.yaml`

### 3.5 Split requirements ✅

- [x] `requirements/base.txt`, `requirements/train.txt`, `requirements/api.txt`, `requirements/ui.txt`, `requirements/dev.txt` created
- [x] Root `requirements.txt` is now a shim pointing to `requirements/dev.txt`
- [x] `Dockerfile.api` installs from `requirements/api.txt`
- [x] `Dockerfile.ui` installs from `requirements/ui.txt`
- [ ] `Dockerfile.dev` — update to use `requirements/train.txt` (currently installs inline; deferred)
- [ ] CI `test` job — currently installs root `requirements.txt` (shim → dev.txt); acceptable for now but should install `requirements/dev.txt` directly in future

### 3.6 Fix .gitignore ✅

- [x] Added: `**/.ipynb_checkpoints/`, `.Trash-0/`, `**/.Trash-0/`, `.venv/`, `mlruns/`, `mlartifacts/`, `artifacts/`, `.DS_Store`, `.env`, `__pycache__/`, `*.py[cod]`, `*.egg-info/`, `build/`, `dist/`, `.pytest_cache/`, `.mypy_cache/`, `*.postman_environment.json`/`!*.example.json`, model artifact patterns under `notebooks/`
- [x] Rewrote `.gitattributes` to enforce LF for all text files; `.bat`/`.ps1` keep CRLF; added `*.pdf`/`*.png` as binary

### 3.7 Notebook filename cleanup ✅

- [x] `04 Experimentation.ipynb` → `04_Experimentation.ipynb`
- [x] `AWS Architecture.ipynb` → `AWS_Architecture.ipynb`

### 3.8 Remove untracked clutter ✅

- [x] `notebooks/.Trash-0/` deleted
- [x] `notebooks/ids_unsw/.ipynb_checkpoints/` deleted
- [ ] `notebooks/ids_unsw/05_Deployment_Optimization2.ipynb` — keep for now (has content); user to decide
- [ ] PDF/PNG exports in `notebooks/ids_unsw/` (`fig_*.pdf`, `mlops_eks_architecture.*`) — move to `Paper/` or delete; user to decide

---

## Pass 4 — Confidence Cleanup

**Goal:** Tests, validation, and CI actually prove what they claim to prove.

### 4.1 Bundle validation script

- [ ] Create `ids_unsw/validate_bundle.py` (or `tools/validate_bundle.py`) that:
  - Loads `xgb.onnx` and verifies input shape matches `feature_names.json` count
  - Parses `metadata.json` against the canonical schema (validates required keys, types, `schema_version`)
  - Loads `scaler.pkl` and verifies it transforms the correct feature count
  - Exits non-zero with a clear message on any failure
- [ ] Call this script in API startup so bad bundles fail fast
- [ ] Add it as a CI step before the Docker build

### 4.2 Improve unit tests

File: `tests/test_bundle.py`

- [ ] Add test: `metadata.json` has `schema_version` key
- [ ] Add test: `metadata.json` does NOT contain legacy keys (`"rf"`, `"xgboost"` nested block, `"champion"`)
- [ ] Add test: `metadata.json` `metrics_at_threshold` keys are all lowercase (`fpr`, not `FPR`)
- [ ] Add test: threshold is a float in range (0, 1)
- [ ] Add test: ONNX model input count matches `feature_names.json` length
- [ ] Add test: `/health` endpoint returns 200 without any `Authorization` header

### 4.3 API integration tests

Create `tests/test_api_integration.py`:

- [ ] `/health` returns 200 with no auth header
- [ ] `/health` response body includes `{"status": "ok"}`
- [ ] `/predict` returns 401 with no auth header
- [ ] `/predict_proba` returns 401 with no auth header
- [ ] `/features` returns 401 with no auth header
- [ ] `/features` returns 200 with valid Bearer token and a list of 34 feature names
- [ ] `/predict` with valid payload and valid token returns 200 with expected shape
- [ ] `/set_threshold` persists the new value (verified by subsequent `/metadata` or `/health` call)
- [ ] Invalid payload to `/predict` returns 422

### 4.4 UI smoke test

- [ ] Add a minimal CI step or script `tools/ui_smoke.py` that:
  - Starts the Dash app in a subprocess
  - Checks that `http://localhost:8050` returns HTTP 200
  - Exits 0 on success, 1 on failure

### 4.5 Manifest validation in CI

File: `.github/workflows/ci.yml`

- [ ] Add a job `manifest-lint` that runs:
  ```bash
  kubectl kustomize k8s/base
  kubectl kustomize k8s/overlays/dev
  kubectl kustomize k8s/overlays/eks-dev
  ```
  and optionally pipes output through `kubeconform` or `kubectl apply --dry-run=client`
- [ ] This job should fail the workflow if any overlay fails to render

### 4.6 Label CI synthetic jobs honestly

File: `.github/workflows/ci.yml`

- [ ] Add a top-of-file comment block explaining that the `data`, `features`, and `train` jobs are pipeline-shape smoke tests, not real model training
- [ ] Rename the train job's step from "Train & log with MLflow" to "Smoke: synthetic train (pipeline contract only)"

---

## Pass 5 — Narrative Cleanup

**Goal:** Paper, docs, and README describe the real system.
Do this pass last — after the repo is stable.

### 5.1 Add architecture document

- [ ] Create `docs/architecture.md` with:
  - System diagram (ASCII or embedded PNG)
  - Description of each layer (data → training → ONNX → API → UI → K8s)
  - Artifact flow: how a bundle gets from a notebook export to the running API
  - Deployment flow: how CI/CD pushes to EKS

### 5.2 Paper alignment

File: `Paper/Nexusnode_MLOPS_Paper/template.tex`

Verify and fix:

- [ ] All API paths match reality (`/health`, `/predict`, `/predict_proba`, `/set_threshold`, `/reload`)
- [ ] Auth description: `/health` is public; scoring/admin require Bearer token — remove any claim that `/health` requires auth
- [ ] UI description says **Dash**, not Streamlit
- [ ] Repository references use `UNSW-NB15_MLOPS`
- [ ] Bundle format description matches canonical schema (`xgb.onnx`, `feature_names.json`, `metadata.json`)
- [ ] Deployment description matches Kubernetes/EKS path
- [ ] Remove any references to legacy file paths (`app.py` at root, `app_dash.py` at root)
- [ ] Any performance metrics cited match `metrics_at_threshold` in the production bundle

### 5.3 README final pass

File: `README.md`

- [ ] Quick Start works end-to-end after a fresh clone (verify manually or in CI)
- [ ] Compose section accurately describes all three compose files and their purposes
- [ ] Configuration table includes all five canonical env vars including `IDS_EXPOSE_DOCS`
- [ ] Add repo badges: CI status, Docker Hub image, license
- [ ] Add a "Changelog / Release Notes" link or inline section noting current version

### 5.4 Add CHANGELOG

- [ ] Create `CHANGELOG.md` with an initial entry describing the current state after cleanup:
  - Schema version 1.0
  - Champion model: XGBoost / ONNX
  - Serving: FastAPI + onnxruntime
  - UI: Dash
  - Deploy: Kubernetes/EKS

---

## Definition of Done

The cleanup is complete only when **all** of the following are true:

**External trust checks (a reviewer or recruiter can verify):**

- [ ] `git clone` + following README produces a working API in one documented command
- [ ] `docker compose -f docker-compose.serve.yml up` starts the API and UI
- [ ] `pytest -q` passes
- [ ] `kubectl kustomize k8s/overlays/eks-dev` renders without errors
- [ ] No resource in the live manifests is named `streamlit`
- [ ] No command in the README says `uvicorn app:app`
- [ ] No command in the README implies `/health` requires auth

**Internal truth checks:**

- [ ] README matches `ids_unsw/serve/app.py` behavior
- [ ] Dockerfiles use only canonical env var names
- [ ] `k8s/base/` contains `dash-ui.yaml`, not `streamlit.yaml`
- [ ] `metadata.json` in the production bundle has `schema_version` and no legacy comparison blocks
- [ ] `docker-compose.yml` does not exist — the three split files do
- [ ] `.gitignore` covers `.ipynb_checkpoints/`, `.Trash-0/`, `.venv/`, `*.pkl` (notebooks), `*.onnx` (notebooks)
- [ ] Paper describes the system that exists, not the system that existed

---

## File Change Summary

Quick reference of every file this plan touches:

| File | Pass | Action |
|---|---|---|
| `README.md` | 1, 2, 3, 5 | Multiple truth fixes |
| `docker-compose.yml` | 1, 3 | Rename service; rename file to `docker-compose.dev.yml` |
| `docker-compose.mlflow.yml` | 3 | Rename to `docker-compose.mlflow.yml` (if needed) |
| `docker-compose.serve.yml` | 3 | **Create new** |
| `.github/workflows/ci.yml` | 1, 4 | Rename jobs/steps; add manifest lint job |
| `.github/workflows/deploy.yml` | 1 | Update name and image slug |
| `ids_unsw/serve/app.py` | 2 | Auth contract, `IDS_EXPOSE_DOCS`, startup validation |
| `notebooks/ids_unsw/models/bundle_xgb/metadata.json` | 2 | Replace with canonical schema |
| `notebooks/ids_unsw/models/metadata.json` | 2 | Replace with canonical schema |
| `Dockerfile.api` | 3 | Env var alignment |
| `Dockerfile.ui` | 3 | Env var alignment; resolve duplicate |
| `Dockerfile.dev` | 3 | Remove TF identity residue |
| `ids_unsw/ui/Dockerfile.ui` | 3 | Decide canonical vs duplicate |
| `k8s/base/streamlit.yaml` | 3 | Rename → `dash-ui.yaml`; update Deployment/Service names |
| `k8s/base/kustomization.yaml` | 3 | Update resource list |
| `k8s/overlays/dev/streamlit-resources.yaml` | 3 | Rename → `dash-ui-resources.yaml` |
| `k8s/overlays/dev/kustomization.yaml` | 3 | Update patches target |
| `k8s/overlays/eks-dev/streamlit-resources.yaml` | 3 | Rename → `dash-ui-resources.yaml` |
| `k8s/overlays/eks-dev/kustomization.yaml` | 3 | Update patches target |
| `k8s/system/metrics-server/kustomization2.yaml` | 3 | Rename → `kustomization.yaml` |
| `requirements.txt` | 3 | Refactor into `requirements/` split |
| `.gitignore` | 3 | Add missing patterns |
| `.gitattributes` | 3 | Verify LF + binary rules |
| `notebooks/ids_unsw/04 Experimentation.ipynb` | 3 | Rename (remove space) |
| `notebooks/ids_unsw/AWS Architecture.ipynb` | 3 | Rename (remove space) |
| `tests/test_bundle.py` | 4 | Expand coverage |
| `tests/test_api_integration.py` | 4 | **Create new** |
| `ids_unsw/validate_bundle.py` | 4 | **Create new** |
| `Paper/Nexusnode_MLOPS_Paper/template.tex` | 5 | Align with real implementation |
| `docs/architecture.md` | 5 | **Create new** |
| `CHANGELOG.md` | 5 | **Create new** |
