# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-03-22

Initial production-ready release following comprehensive operational cleanup.

### Added
- **Canonical Metadata Schema**: Schema version 1.0 enforced for bundle `metadata.json` ensuring no leakage or legacy artifacts.
- **Strict Bundle Validation**: `ids_unsw.validate_bundle` runs as an end-to-end integrity gate gating API startup and CI building. 
- **API Test Coverage**: In-process FastApi test client (`test_api_integration.py`) added to guarantee auth constraints and shape requirements.
- **Dash UI E2E Smoke Tests**: Local Dash verification added conditionally testing component loads via a subprocess (`test_ui_smoke.py`). 
- **Kubernetes Architecture**: Scalable, public/private EKS-compatible overlays (`dash-ui` public ingress vs `ids-api` internal clusterIP service configuration) with metrics-driven CPU HPA.
- **CI/CD Manifest Validation**: Integrated `kubectl kustomize` passes across the pipeline block syntax regressions.
- **EKS System Architecture Docs**: Defined `docs/architecture.md` and archived smoke-test evidence under `docs/archive/eks-smoke-test-results.md`.

### Changed
- **Documentation layout**: Completed cleanup checklist (`cleanup-plan.md`) and the point-in-time EKS smoke-test report moved to `docs/archive/` with an index (`docs/archive/README.md`).
- **Unified Naming Scheme**: Repository converted entirely to `UNSW-NB15_MLOPS`; `tensorflow` dev containers shifted to `mlops-dev`.
- **Requirements Structure**: Base requirements effectively split across `api.txt`, `ui.txt`, `dev.txt`, and `train.txt` to minimize dependencies per container profile.
- **API Health Probes (`/health`)**: Removed authorization headers from K8s manifest probes—exempted `/health` route entirely from auth, ensuring probe stability.
- **UI Service Migrated**: Core visual interaction transitioned completely from legacy Streamlit frameworks onto a unified `gunicorn` initialized Plotly Dash instance (`ids_unsw/ui/app_dash.py`).
- **Compose Purpose Context**: Local compose structures renamed `docker-compose.dev.yml`, `docker-compose.serve.yml`, and `docker-compose.mlflow.yml` detailing operations.

### Removed
- Deprecated unreferenced frontend code.
- Removed legacy comparison models and keys (`rf`, `xgboost.champion`) nested inside the production `metadata.json` registry.
