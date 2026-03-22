# Architecture Decisions — UNSW-NB15_MLOPS

This file is the canonical referee for every edit in this repository.
When any README, Dockerfile, manifest, or workflow contradicts something written here, this file wins.

---

## Locked Decisions

### Project & Repository Identity

| Concern | Decision |
|---|---|
| Display / project name | `UNSW-NB15_MLOPS` |
| Ops / infra slug (Docker, K8s labels, image names) | `unsw-nb15-mlops` |
| Python package | `ids_unsw` — **do not rename** |
| Repository URL slug | `UNSW-NB15_mlops` (existing GitHub name, unchanged) |

The rule is simple:

- Anywhere a human reads a title, header, or doc → `UNSW-NB15_MLOPS`
- Anywhere a tool, registry, or manifest uses a name → `unsw-nb15-mlops`
- Anywhere Python imports → `ids_unsw`

### Stack Identity

| Layer | Technology |
|---|---|
| Champion model | XGBoost, exported to ONNX |
| Inference runtime | `onnxruntime` |
| Serving API | FastAPI (`ids_unsw/serve/app.py`) |
| UI | Dash (`ids_unsw/ui/app_dash.py`) — **not Streamlit** |
| Deployment target | Kubernetes / EKS via Kustomize |
| Experiment tracking | MLflow (optional, separate stack) |

Streamlit is a **retired/legacy** UI. No active resource, manifest, or document should describe the current UI as Streamlit.

### Serving Contract (Auth)

| Endpoint class | Auth requirement |
|---|---|
| `/health` | **Public** — no Bearer token required |
| `/features`, `/metadata` | Bearer token required |
| `/predict`, `/predict_proba` | Bearer token required |
| `/set_threshold`, `/reload`, `/deploy_registry` | Bearer token required |
| `/docs`, `/openapi.json` | Controlled by `IDS_EXPOSE_DOCS` env var |

`/health` is **intentionally unauthenticated**. Any README, paper, or probe config claiming it requires auth is wrong.

### Canonical Environment Variables

| Variable | Used by | Purpose |
|---|---|---|
| `IDS_API_TOKEN` | API, UI, K8s | Bearer token for auth |
| `IDS_BUNDLE_DIR` | API | Directory containing `xgb.onnx`, `feature_names.json`, `metadata.json` |
| `IDS_SCALER_PATH` | API | Path to `scaler.pkl` |
| `IDS_API_URL` | Dash UI | Base URL of the API |
| `IDS_EXPOSE_DOCS` | API | `true`/`false` — controls `/docs` and `/openapi.json` exposure |
| `MLFLOW_TRACKING_URI` | Training / experiments | Optional MLflow URI |

No other names for these variables are canonical. If Dockerfiles, manifests, or docs use different names for the same concepts, they are wrong.

### Canonical Artifact Bundle Schema

The live-serving bundle must contain exactly:

```
bundle_xgb/
  xgb.onnx
  feature_names.json
  metadata.json
  scaler.pkl           (or referenced via IDS_SCALER_PATH outside the bundle)
```

The canonical `metadata.json` schema is:

```json
{
  "schema_version": "1.0",
  "project_name": "UNSW-NB15_MLOPS",
  "model_family": "xgboost",
  "artifact_format": "onnx",
  "n_features": 34,
  "threshold": 0.757143,
  "feature_source": "feature_names.json",
  "metrics_at_threshold": {
    "precision": 0.9381473409872161,
    "recall":    0.9211153269213801,
    "f1":        0.9295533219799423,
    "roc_auc":   0.9805389430174021,
    "fpr":       0.07440540540540541,
    "tp":        41756,
    "fp":        2753,
    "tn":        34247,
    "fn":        3576
  },
  "training_context": {
    "dataset": "UNSW-NB15",
    "recall_min_constraint": 0.95
  }
}
```

**What must be removed from production metadata:**

- The `"rf"` block (old Random Forest comparison)
- The `"xgboost"` nested block (stale experiment-phase threshold and metrics)
- The `"champion"` field (redundant — project name says it)
- The `"features"` inline array (belongs in `feature_names.json`, not `metadata.json`)

Model comparison history belongs in MLflow, notebook exports, or evaluation artifacts — not in the production bundle.

### Compose File Responsibilities

| File | Purpose |
|---|---|
| `docker-compose.dev.yml` | GPU-enabled dev/notebook environment (JupyterLab) |
| `docker-compose.serve.yml` | Local API + Dash UI serving stack |
| `docker-compose.mlflow.yml` | Optional local MLflow tracking server |

The current `docker-compose.yml` is the dev/notebook environment. It should be renamed to `docker-compose.dev.yml`.

### Kubernetes Layout

```
k8s/
  base/
    kustomization.yaml
    ids-api.yaml          (was/is correct)
    dash-ui.yaml          (rename from streamlit.yaml)
  overlays/
    local/
    dev/
    eks-dev/
  system/
    metrics-server/
      kustomization.yaml  (fix from kustomization2.yaml)
```

No active resource should use the name `streamlit`. All Deployment/Service objects named `streamlit` must be renamed `dash-ui`.

### Dependency Structure

| File | Contains |
|---|---|
| `requirements/base.txt` | numpy, pandas, scikit-learn |
| `requirements/train.txt` | xgboost, torch, mlflow, onnx conversion |
| `requirements/api.txt` | fastapi, uvicorn, onnxruntime, pydantic |
| `requirements/ui.txt` | dash, dash-bootstrap-components, requests, gunicorn |
| `requirements/dev.txt` | pytest, black, ruff, mypy, pre-commit |

The current single `requirements.txt` may remain as a convenience shim that `pip install`s the split files, or it becomes the api+base install for backward compatibility during transition.

### CI Pipeline Contract

| Job | What it actually proves |
|---|---|
| `lint` | ruff + black --check |
| `test` | pytest bundle/unit tests |
| `data` / `features` / `train` | **Smoke fixtures only** — tiny synthetic pipeline to validate CI plumbing, not real training |
| `docker` | API image builds, boots, passes integration checks |
| `deploy` | Manifest validation + rollout + post-deploy `/health` |

CI synthetic training jobs must be labelled as smoke fixtures, not presented as real training.

---

## Anti-Patterns to Eliminate

These are wrong everywhere and should be removed:

- Service, container, or image named `tensorflow` or `tf-gpu` — replace with `mlops-dev` or `dev-notebook`
- Image tag `tf-gpu-fixed` — replace with `unsw-mlops-dev` or similar
- `uvicorn app:app` — replace with `uvicorn ids_unsw.serve.app:app`
- Root-level `app.py` / `app_dash.py` references in docs — they live at `ids_unsw/serve/app.py` and `ids_unsw/ui/app_dash.py`
- Any claim that `/health` requires a Bearer token
- Active K8s objects named `streamlit-*`
- `kustomization2.yaml` — non-standard filename that Kustomize ignores
- Nested `"rf"` / `"xgboost"` comparison blocks in production `metadata.json`
- Notebook output directories (`.ipynb_checkpoints`, `.Trash-0`) tracked in git
- Spaces in notebook filenames (`04 Experimentation.ipynb`, `AWS Architecture.ipynb`)
