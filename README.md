# UNSW-NB15_MLOPS

[![CI](https://github.com/vincembanze/UNSW-NB15_mlops/actions/workflows/ci.yml/badge.svg)](https://github.com/vincembanze/UNSW-NB15_mlops/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
**Version 1.0** — [See release notes in CHANGELOG.md](./CHANGELOG.md)

End-to-end, production-style MLOps pipeline for **network intrusion detection** on the **UNSW-NB15** dataset.
It covers the full stack: **data engineering → model training (XGBoost) → threshold selection → ONNX export → FastAPI service → Dash UI → CI smoke tests → Kubernetes (HPA, probes, kustomize).**

> **TL;DR**
>
> * Train an XGBoost model, export to **ONNX**, pick a serving **threshold**, and bundle artifacts.
> * Serve with **FastAPI** (`/predict`, `/predict_proba`, `/health`, `/set_threshold`, …).
> * Control from a minimal **Dash UI**.
> * Ship to **Kubernetes** with **readiness/liveness** probes and **HPA**.
> * Test locally via **Postman collection** (environment provided as example only).

---

## Table of Contents

* [Project Goals](#project-goals)
* [Architecture](#architecture)
* [Repo Layout](#repo-layout)
* [Quick Start](#quick-start)

  * [A. Local (Python)](#a-local-python)
  * [B. Docker Compose](#b-docker-compose)
  * [C. MLflow (optional)](#c-mlflow-optional)
* [Training → ONNX → Bundle](#training--onnx--bundle)
* [Serving API (FastAPI)](#serving-api-fastapi)
* [Dash UI](#dash-ui)
* [Kubernetes (kustomize, HPA, probes)](#kubernetes-kustomize-hpa-probes)
* [Testing](#testing)

  * [pytest bundle checks](#pytest-bundle-checks)
  * [Postman](#postman)
* [Configuration & Environment Variables](#configuration--environment-variables)
* [Security Notes](#security-notes)
* [Data Source & Licensing](#data-source--licensing)
* [Troubleshooting](#troubleshooting)
* [Contributing](#contributing)

---

## Project Goals

1. **Reproducible ML pipeline** for an IDS classifier on UNSW-NB15.
2. **Portable inference** via **ONNX** and **onnxruntime**.
3. **Operational controls**: explicit operating **threshold**, **health checks**, and **autoscaling** readiness.
4. **Dev→Prod path**: local, Docker, and **Kubernetes** with **HPA**.

---

## Architecture

```
┌─────────────────────────┐     ┌─────────────────────────┐
│  Feature Engineering    │     │     Model Training      │
│  (encode/clean/dedupe)  │──►──│  XGBoost + metrics      │
│  + feature_names.json   │     │  MLflow (optional)      │
└─────────────────────────┘     └──────────┬──────────────┘
                                            │
                                            ▼
                           ┌──────────────────────────────────┐
                           │ Threshold Selection (xgb_* tools)│
                           │  → metadata.json (threshold, fx) │
                           └─────────────────┬────────────────┘
                                             │
                                             ▼
                         ┌────────────────────────────────────┐
                         │     ONNX Export (xgb.onnx)         │
                         │  Bundle: {xgb.onnx, feature_names, │
                         │           metadata.json, scaler}   │
                         └─────────────────┬──────────────────┘
                                           │
                                           ▼
                ┌─────────────────────────────────────────────┐
                │  FastAPI (ids_unsw/serve/app.py)           │
                │  Endpoints: /health /features /predict*    │
                │  Auth: Bearer IDS_API_TOKEN                │
                └───────────────┬────────────────────────────┘
                                │
                                ▼
                     ┌───────────────────────┐
                     │ Dash UI (app_dash.py) │
                     │  for scoring & ops    │
                     └───────────────────────┘
```

---

## Repo Layout

```
UNSW-NB15_mlops/
├─ ids_unsw/
│  ├─ features/                # feature engineering scripts
│  ├─ experiments/             # xgb_to_onnx, threshold, registry helpers
│  ├─ serve/
│  │  └─ app.py                # FastAPI serving entrypoint
│  ├─ ui/
│  │  └─ app_dash.py           # Dash UI entrypoint
│  └─ tests/                   # Postman collection (env example only)
├─ notebooks/
│  └─ ids_unsw/
│     ├─ data/                 # parquet inputs/outputs (not in git)
│     └─ models/               # artifacts (not in git)
│        └─ bundle_xgb/        # xgb.onnx, feature_names.json, metadata.json
├─ tests/                      # pytest bundle tests
├─ k8s/                        # kustomize manifests (api, ui, hpa, patches)
├─ docs/                       # architecture decisions; archive/ for completed checklists & reports
├─ docker-compose.dev.yml      # GPU dev/notebook environment (JupyterLab)
├─ docker-compose.serve.yml    # local API + Dash UI serving stack
├─ docker-compose.mlflow.yml   # optional local MLflow tracking server
├─ requirements.txt
└─ README.md
```

> Your actual tree may differ slightly—this README assumes the common paths used in the scripts and notebooks.

---

## Quick Start

### A. Local (Python)

> Requires **Python 3.11+**.

```bash
# Linux/macOS (bash)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

```powershell
# Windows (PowerShell)
python -m venv .venv
. .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Place (or generate) your bundle here:

```
notebooks/ids_unsw/models/bundle_xgb/
  ├─ xgb.onnx
  ├─ feature_names.json
  └─ metadata.json           # includes "threshold"
notebooks/ids_unsw/models/scaler.pkl
```

Run the API:

```bash
export IDS_API_TOKEN="REPLACE_WITH_LONG_RANDOM"
export IDS_BUNDLE_DIR="notebooks/ids_unsw/models/bundle_xgb"
export IDS_SCALER_PATH="notebooks/ids_unsw/models/scaler.pkl"
uvicorn ids_unsw.serve.app:app --host 0.0.0.0 --port 8000
```

```powershell
$env:IDS_API_TOKEN="REPLACE_WITH_LONG_RANDOM"
$env:IDS_BUNDLE_DIR="notebooks/ids_unsw/models/bundle_xgb"
$env:IDS_SCALER_PATH="notebooks/ids_unsw/models/scaler.pkl"
uvicorn ids_unsw.serve.app:app --host 0.0.0.0 --port 8000
```

Smoke it (`/health` is public — no token needed):

```bash
curl http://localhost:8000/health
```

### B. Docker Compose

Three compose files, each with a single purpose:

| File | Purpose |
|---|---|
| `docker-compose.dev.yml` | GPU-enabled JupyterLab for training/experiments |
| `docker-compose.serve.yml` | Local API + Dash UI serving stack |
| `docker-compose.mlflow.yml` | Optional local MLflow tracking server |

**Local serving stack** (API + Dash UI):

```bash
# .env
IDS_API_TOKEN=REPLACE_WITH_LONG_RANDOM
IDS_BUNDLE_DIR=notebooks/ids_unsw/models/bundle_xgb
IDS_SCALER_PATH=notebooks/ids_unsw/models/scaler.pkl
```

```bash
docker compose -f docker-compose.serve.yml up --build
# API  : http://localhost:8000
# DASH : http://localhost:8050
```

**Dev/notebook environment** (GPU required):

```bash
docker compose -f docker-compose.dev.yml up --build
# JupyterLab: http://localhost:8888
```

### C. MLflow (optional)

A convenience Compose stack for **local** MLflow is provided:

```bash
docker compose -f docker-compose.mlflow.yml up -d
export MLFLOW_TRACKING_URI="http://localhost:5000"
```

---

## Training → ONNX → Bundle

1. **Feature engineering** (parquet in → clean parquet + preprocessor):

```bash
python ids_unsw/features/engineer.py \
  --train-input  notebooks/ids_unsw/data/UNSW_NB15_training-set.parquet \
  --test-input   notebooks/ids_unsw/data/UNSW_NB15_testing-set.parquet \
  --train-output notebooks/ids_unsw/data/UNSW_NB15_train_clean.parquet \
  --test-output  notebooks/ids_unsw/data/UNSW_NB15_test_clean.parquet \
  --preprocessor-out notebooks/ids_unsw/models/cat_preprocessor.pkl
```

2. **Train** (XGBoost is the default champion; logs to MLflow if configured):

```bash
export MLFLOW_TRACKING_URI=http://localhost:5000   # optional
python ids_unsw/experiments/train.py \
  --train-input notebooks/ids_unsw/data/UNSW_NB15_train_clean.parquet \
  --test-input  notebooks/ids_unsw/data/UNSW_NB15_test_clean.parquet \
  --models-dir  notebooks/ids_unsw/models \
  --mlflow-uri  "$MLFLOW_TRACKING_URI" \
  --mlflow-exp  unsw-nb15 \
  --train-xgb --xgb-rounds 400 --xgb-early 50 \
  --save-artifacts
```

3. **Export to ONNX**:

```bash
python ids_unsw/experiments/xgb_to_onnx.py --base notebooks/ids_unsw
# → notebooks/ids_unsw/models/xgb.onnx + feature_names.json
```

4. **Pick & persist threshold** (write `metadata.json`):

```bash
python ids_unsw/experiments/xgb_persist.py \
  --base notebooks/ids_unsw \
  --mlflow-uri "$MLFLOW_TRACKING_URI" \
  --mlflow-exp unsw-nb15 \
  --recall-min 0.95
# → metadata.json has {"threshold": ...} and summary metrics
```

5. **(Optional)** Register ONNX in MLflow Model Registry:

```bash
python ids_unsw/experiments/register_onnx.py \
  --mlflow-uri "$MLFLOW_TRACKING_URI" \
  --mlflow-exp unsw-nb15 \
  --base notebooks/ids_unsw \
  --name unsw_xgb_ids_onnx
```

---

## Serving API (FastAPI)

**Run:**

```bash
export IDS_API_TOKEN="REPLACE"
export IDS_BUNDLE_DIR="notebooks/ids_unsw/models/bundle_xgb"
export IDS_SCALER_PATH="notebooks/ids_unsw/models/scaler.pkl"
uvicorn ids_unsw.serve.app:app --host 0.0.0.0 --port 8000
```

**Endpoints:**

* `GET /health` → status + threshold — **public, no auth required**
* `GET /features` → list of feature names — Bearer auth required
* `GET /metadata` → bundle metadata — Bearer auth required
* `POST /predict_proba` → returns probabilities — Bearer auth required
* `POST /predict` → returns class labels given the current threshold — Bearer auth required
* `POST /set_threshold` → update threshold in memory (and persist) — Bearer auth required
* `POST /reload` → reload artifacts from bundle dir — Bearer auth required
* `POST /deploy_registry` → (optional) pull new model from MLflow registry — Bearer auth required

**JSON example:**

```bash
curl -X POST http://localhost:8000/predict_proba \
  -H "Authorization: Bearer $IDS_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"instances":[{"feat1":0,"feat2":0,"...":"..."}]}'
```

---

## Dash UI

Run locally:

```bash
export IDS_API_URL="http://localhost:8000"
export IDS_API_TOKEN="REPLACE"
python ids_unsw/ui/app_dash.py
# http://localhost:8050
```

It fetches `/features` to build an input form, lets you **score** rows/CSV, tweak **threshold**, and call **/reload**.

---

## Kubernetes (kustomize, HPA, probes)

This repo ships **kustomize** resources and patches for a token-protected API:

* **Deployments/Services** for API and UI
* **Readiness/Liveness** probes using `curl /health` — no auth required on this endpoint
* **HPA** (autoscaling/v2) on CPU (or extend to custom metrics)
* **metrics-server** Service patch (common port/name mismatch fix)

**Secrets/Config (example in kustomization):**

* Secret `ids-api-secrets`: `IDS_API_TOKEN`
* ConfigMap `ids-api-config`: `IDS_BUNDLE_DIR`, `IDS_SCALER_PATH`, `IDS_API_URL`

**Apply:**

```bash
kubectl apply -k k8s/overlays/dev
kubectl get pods,svc,hpa
kubectl top pods -A   # if metrics-server is patched and working
```

> Bundle artifacts must be accessible in the container (bake into image or mount a read-only PVC).

---

## Testing

### pytest bundle checks

A small test ensures your **bundle** is consistent before deployment:

* `xgb.onnx` loads
* `feature_names.json` has the expected count (e.g., **34**)
* `metadata.json` contains `"threshold"`

Run:

```bash
pip install -r requirements.txt
pytest -q
```

### Postman

* Collection: `ids_unsw/tests/UNSW_IDS_API.postman_collection.json`
* **Do not commit real environments.** Instead, provide an example:
  `ids_unsw/tests/UNSW_IDS_API.postman_environment.example.json`

  ```json
  {
    "name": "Local Example",
    "values": [
      { "key": "baseUrl",  "value": "http://localhost:8000", "enabled": true },
      { "key": "apiToken", "value": "REPLACE_ME", "enabled": true }
    ]
  }
  ```

Import the **collection**, **duplicate** the example env, insert your token, and run the suite.

---

## Configuration & Environment Variables

| Variable              | Where         | Description                                                       |
| --------------------- | ------------- | ----------------------------------------------------------------- |
| `IDS_API_TOKEN`       | API, UI, K8s  | **Required** Bearer token — all endpoints except `/health`        |
| `IDS_BUNDLE_DIR`      | API           | Directory with `xgb.onnx`, `feature_names.json`, `metadata.json` |
| `IDS_SCALER_PATH`     | API           | Path to `scaler.pkl`                                              |
| `IDS_API_URL`         | Dash UI       | Base URL of API (e.g., `http://ids-api:8000`)                     |
| `IDS_EXPOSE_DOCS`     | API           | `true`/`false` — controls `/docs` and `/openapi.json` exposure    |
| `MLFLOW_TRACKING_URI` | Training/Exp. | Optional MLflow server URI (e.g., `http://localhost:5000`)        |

**.gitignore** includes env/secrets patterns. Keep **real tokens out of git**.

---

## Security Notes

* **Never** commit real Postman environments or `.env` files.
* Rotate any leaked tokens; if something was committed, **rewrite history** (done here) and force-push.
* In K8s, use `Secret` for `IDS_API_TOKEN`; for GitOps, consider **SOPS** to encrypt.
* Avoid deploying the exact same thresholds and artifact bundle you open-source if you care about reducing information disclosure (parametrize in prod).

---

## Data Source & Licensing

* **Dataset:** UNSW-NB15 (UNSW Canberra Cyber Range Lab).
  You can obtain it via the official UNSW source or mirrored datasets (e.g., Kaggle).
  Respect the **license/terms** provided by the dataset owner(s).
* **Repo license:** choose one that matches your goals; **Apache-2.0** is a good default for permissive reuse.

---

## Troubleshooting

* **`/health` returns 401/403** → `/health` is intentionally public and should never require auth. If it returns 401, there is a middleware misconfiguration — check that no global auth dependency wraps all routes.
* **`onnxruntime` errors** → check that your `xgb.onnx` matches the trained model and your `feature_names.json` ordering.
* **Wrong feature count** → regenerate `feature_names.json` to match engineering, or update tests.
* **HPA doesn’t scale / `kubectl top` empty** → ensure `metrics-server` is running; apply the `metrics-server` service patch if needed.
* **Windows paths** → prefer forward slashes in config or quote paths with spaces in PowerShell.

---

## Contributing

1. Fork and create a feature branch.
2. Keep secrets out of commits.
3. Add/extend tests (`pytest`), and update docs if paths change.
4. Open a PR with a clear description of changes and validation steps.


