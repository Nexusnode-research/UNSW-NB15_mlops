# UNSW-NB15_MLOPS Architecture

This document describes the end-to-end architecture of the machine learning pipeline, artifact flow, and cloud deployment environment.

## 1. System Diagram

```mermaid
flowchart TD
    subgraph Data & Training ["Data & Model Training (Offline)"]
        Raw[Raw Parquet] --> Engineer[ids_unsw/features/engineer.py]
        Engineer --> Clean[Clean Parquet]
        Engineer --> Scaler[scaler.pkl]
        Clean --> Train[train.py (XGBoost)]
        Train --> Model[xgb.json]
    end

    subgraph Evaluation ["Evaluation & Bundle (Offline)"]
        Model --> Threshold[xgb_persist.py]
        Clean --> Threshold
        Threshold --> Meta[metadata.json]
        Model --> Export[xgb_to_onnx.py]
        Export --> ONNX[xgb.onnx]
        Export --> Features[feature_names.json]
    end

    subgraph Artifact Flow ["Artifact Bundle"]
        ONNX --> Bundle[Deployable Bundle]
        Features --> Bundle
        Meta --> Bundle
        Scaler --> Bundle
    end

    subgraph Serving Environment ["K8s Serving Stack (API + UI)"]
        Bundle --> API[ids_unsw/serve/app.py (FastAPI)]
        API <--> UI[ids_unsw/ui/app_dash.py (Dash)]
    end

    subgraph End Users
        UI <--> User[Operator / Web Client]
        HTTPClient[API Client] <--> API
    end
```

## 2. Layer Description

### 2.1 Data (Engineering)
Raw UNSW-NB15 parquet files are processed by `engineer.py`. Categorical features (e.g., `proto`, `service`, `state`) are encoded deterministically with fallback mechanisms for unseen values (`-1`). Output artifacts include the fully numeric, float32 clean parquet files, the fitted `cat_preprocessor.pkl`, and the training scaler (`scaler.pkl`).

### 2.2 Training
A champion XGBoost classifier is trained on the parsed feature matrix treating the data class balance via `scale_pos_weight`. Model hyperparameters are tuned via random search with early stopping on a validation split, yielding a binary logistic output format. Parameters, metrics, and models can optionally trace to an MLflow back-end.

### 2.3 ONNX & Metadata (Thresholding)
The finalized XGBoost artifact is converted to ONNX standard form using `onnxmltools` (`skl2onnx`), producing an implementation-independent graph `xgb.onnx` returning an `[N, 2]` probability tensor.
Rather than hardcoding decision bounds, a threshold sweep is conducted (`xgb_persist.py`) to hit operating constraints (e.g., precision/recall trade-offs) and pinned inside a `metadata.json` contract alongside the sequence of features (`feature_names.json`).

### 2.4 API (FastAPI)
A strict schema-contract enforcement wrapper built in FastAPI loads the ONNX runtime session via CPU alongside the `scaler.pkl` state.
* The API requires valid authentication limits via Bearer Tokens before processing predictions (`IDS_API_TOKEN`).
* Before scoring, incoming `.json` batch data is validated to exactly track the 34 features registered in `feature_names.json`.
* Scaling is enacted and inference delegates to `onnxruntime`. The pre-loaded threshold returns hard labels `[0, 1]` on the `/predict` route, while `/predict_proba` returns raw probabilities.

### 2.5 UI (Dash)
Operators govern system thresholds and probe live traffic samples using a Dash UI abstraction (`ids_unsw/ui/app_dash.py`). The internal server fetches schema structures (`/features`) directly from the API and dynamically compiles an application form capable of ad-hoc row analysis.

### 2.6 Kubernetes (K8s) Infrastructure
Containers for the UI (`unsw-nb15-mlops-ui`) and API (`unsw-nb15-mlops-api`) deploy into Elastic Kubernetes Service (EKS) bound to a namespace.
* Internal routing handles communication via a `ClusterIP` DNS resolution mapping UI inference payloads to the internal API Service.
* Pod elasticity utilizes the metrics-server reporting to an active Horizontal Pod Autoscaler (HPA) targeting standard CPU thresholds.
* The workloads map HTTP Readiness and Liveness Probes directly to the public `/health` gateway for automatic rollout progression tracking and instance eviction bounding.

## 3. Artifact Flow (Notebook to Deployment)

The deployment bundle consists intrinsically of immutable files:
1. `xgb.onnx`
2. `feature_names.json`
3. `metadata.json`
4. `scaler.pkl`

When the notebook completes the sweeping tasks and serialization commands, it generates these files inside `notebooks/ids_unsw/models/bundle_xgb`. 
This bundled local state is accessed physically during `docker compose` up-cycle logic via bind mounts OR physically copied into Dockerfile filesystem layers during the `docker build` process for remote CI/CD clusters. The API runs a `validate_bundle.py` startup routine to prevent serving an incomplete filesystem state.

## 4. Deployment Flow (CI/CD to EKS)
Changes pushed to the `main` branch trigger continuous integration workflows on GitHub Actions:
1. **Testing:** `pytest` suites evaluate artifact health, payload formatting, integration edge-cases, and threshold limits.
2. **Dockerization:** Immutable AMD64 Docker images are compiled using deterministic Dockerfiles.
3. **Registry Pipeline:** Images tagged with git-hashes / digests are pushed to an AWS ECR authenticated repository logic pipeline.
4. **Smoke Execution:** The workflow validates kustomize outputs `kubectl kustomize k8s/overlays/*`
5. **EKS Rollout:** `kubectl apply -k k8s/overlays/eks-*` handles iterative application deployments to EKS where `kustomize` logic orchestrates patch rollouts against current operational clusters.
