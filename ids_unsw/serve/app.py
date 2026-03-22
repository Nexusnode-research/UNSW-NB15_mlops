from __future__ import annotations

import os, json, pickle, shutil, glob, tempfile, pathlib
from pathlib import Path
from typing import List, Dict, Any, Optional
from threading import Lock
from contextlib import asynccontextmanager

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException, Body, Depends, Security, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.artifacts import download_artifacts
from ids_unsw.validate_bundle import validate_bundle

# Base/bundle paths (env overrideable)
MODEL_BASE = Path(os.getenv("IDS_BUNDLE_DIR", "notebooks/ids_unsw/models/bundle_xgb"))
SCALER_PATH = Path(os.getenv("IDS_SCALER_PATH", "notebooks/ids_unsw/models/scaler.pkl"))
MODELS = MODEL_BASE.parent

# ---- Docs exposure ----
_expose_docs = os.getenv("IDS_EXPOSE_DOCS", "false").lower() == "true"
_docs_url    = "/docs"        if _expose_docs else None
_openapi_url = "/openapi.json" if _expose_docs else None

_bearer = HTTPBearer(auto_error=False)
def require_token(request: Request, creds: HTTPAuthorizationCredentials = Security(_bearer)):
    if request.url.path == "/health":
        return
    api_token = getattr(request.app.state, "api_token", None)
    if not api_token:
        # In case API token wasn't loaded somehow
        raise HTTPException(status_code=500, detail="Server not configured with API token.")
    if not creds or creds.scheme.lower() != "bearer" or creds.credentials != api_token:
        raise HTTPException(status_code=401, detail="Unauthorized")

_reload_lock = Lock()

def _load_validated_state(bundle_dir: Path, scaler_path: Path) -> dict:
    """Validates the bundle on disk and returns the configured runtime state."""
    errors = validate_bundle(bundle_dir, scaler_path)
    if errors:
        msg = "Bundle validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        raise RuntimeError(msg)

    # accept xgb.onnx or model.onnx
    onnx_path = bundle_dir / "xgb.onnx"
    if not onnx_path.exists():
        onnx_path = bundle_dir / "model.onnx"

    feat_obj = json.loads((bundle_dir / "feature_names.json").read_text())
    features = feat_obj["features"] if isinstance(feat_obj, dict) and "features" in feat_obj else feat_obj

    meta = json.loads((bundle_dir / "metadata.json").read_text())
    thr = float(meta["threshold"])

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inp_name = sess.get_inputs()[0].name
    out_probs_name = sess.get_outputs()[1].name

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    return {
        "features": features,
        "threshold": thr,
        "meta": meta,
        "sess": sess,
        "scaler": scaler,
        "inp_name": inp_name,
        "out_probs_name": out_probs_name,
    }

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load token
    api_token = os.getenv("IDS_API_TOKEN")
    if not api_token:
        raise RuntimeError("IDS_API_TOKEN environment variable is required.")
    app.state.api_token = api_token

    # Load initial model state
    app.state.model_state = _load_validated_state(MODEL_BASE, SCALER_PATH)
    yield
    # Cleanup if needed
    app.state.model_state = None

app = FastAPI(
    title="UNSW-NB15_MLOPS — IDS API",
    version="1.0",
    docs_url=_docs_url,
    openapi_url=_openapi_url,
    dependencies=[Depends(require_token)],
    lifespan=lifespan
)

class PredictRequest(BaseModel):
    instances: List[Dict[str, Any]]

class PredictProbaRequest(BaseModel):
    instances: List[Dict[str, Any]]

class ThresholdIn(BaseModel):
    threshold: float = Field(..., ge=0.0, le=1.0)

class DeployReq(BaseModel):
    model_name: str
    version: str
    tracking_uri: str

def _write_threshold_files(thr: float) -> None:
    """Write threshold to both metadata files (models + bundle)."""
    targets = [MODELS / "metadata.json", MODEL_BASE / "metadata.json"]
    for p in targets:
        if not p.parent.exists(): continue
        try:
            meta = json.loads(p.read_text())
        except Exception:
            meta = {}
        meta["threshold"] = float(thr)
        p.write_text(json.dumps(meta, indent=2))

def _validate_and_stack(rows: List[Dict[str, Any]], features: List[str]) -> np.ndarray:
    """Ensure each row has exactly the expected features and order them."""
    fset = set(features)
    X_list = []
    for i, row in enumerate(rows):
        keys = set(row.keys())
        missing = list(fset - keys)
        extra = list(keys - fset)
        if missing or extra:
            raise HTTPException(
                status_code=422,
                detail={
                    "row_index": i,
                    "error": "feature_mismatch",
                    "missing": missing,
                    "extra": extra,
                    "expected": features,
                },
            )
        ordered = [row[name] for name in features]
        X_list.append(ordered)
    X = np.asarray(X_list, dtype=np.float32)
    return X

@app.get("/health")
def health(request: Request):
    state = getattr(request.app.state, "model_state", None)
    if not state:
        raise HTTPException(status_code=503, detail="Model state not loaded")
    return {
        "status": "ok",
        "model_bundle_dir": str(MODEL_BASE),
        "scaler_path": str(SCALER_PATH),
        "n_features": len(state["features"]),
        "threshold": state["threshold"],
    }

@app.get("/features")
def features(request: Request):
    return {"features": request.app.state.model_state["features"]}

@app.get("/metadata")
def metadata(request: Request):
    return request.app.state.model_state["meta"]

@app.get("/meta")
def meta(request: Request):
    state = request.app.state.model_state
    return {
        "model": "xgboost_onnx",
        "n_features": len(state["features"]),
        "threshold": state["threshold"],
        "meta": state["meta"],
    }

@app.post("/reload")
def reload_model(request: Request):
    with _reload_lock:
        try:
            new_state = _load_validated_state(MODEL_BASE, SCALER_PATH)
            request.app.state.model_state = new_state
            return {
                "ok": True,
                "n_features": len(new_state["features"]),
                "threshold": new_state["threshold"],
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to reload model: {e}")

@app.post("/set_threshold")
def set_threshold(inp: ThresholdIn, request: Request):
    with _reload_lock:
        thr = float(inp.threshold)
        _write_threshold_files(thr)
        try:
            new_state = _load_validated_state(MODEL_BASE, SCALER_PATH)
            request.app.state.model_state = new_state
            live_threshold = new_state["threshold"]
            n_features = len(new_state["features"])
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to reload model after threshold change: {e}")

    return {
        "ok": True,
        "n_features": n_features,
        "threshold": live_threshold,
        "info": {"ok": True, "n_features": n_features, "threshold": live_threshold},
    }

@app.post("/deploy_registry")
def deploy_registry(req: DeployReq, request: Request):
    with _reload_lock:
        mlflow.set_tracking_uri(req.tracking_uri)
        client = MlflowClient()

        tried = []
        def _try(uri: str):
            tried.append(uri)
            return download_artifacts(uri)

        try:
            mv = client.get_model_version(name=req.model_name, version=req.version)
            candidates = [f"models:/{req.model_name}/{req.version}"]
            if getattr(mv, "source", None):
                candidates.append(mv.source)
            if getattr(mv, "run_id", None):
                candidates.append(f"runs:/{mv.run_id}/xgb")

            local = None
            used = None
            errs = []
            for uri in candidates:
                try:
                    local = _try(uri)
                    used = uri
                    break
                except Exception as e:
                    errs.append(f"{uri} -> {e}")

            if local is None:
                raise HTTPException(status_code=400, detail=f"MLflow artifact download failed. Tried:\n" + "\n".join(errs))

            local_dir = Path(local)
            onnx_files = list(local_dir.rglob("*.onnx"))
            if not onnx_files:
                raise HTTPException(status_code=400, detail=f"No .onnx found under downloaded path: {local_dir}")

            # Deploy to a temporary directory to validate first
            with tempfile.TemporaryDirectory() as tmp_str:
                tmp_dir = Path(tmp_str)
                # Copy artifacts to tmp_dir
                for p in local_dir.rglob("*"):
                    rel = p.relative_to(local_dir)
                    dst = tmp_dir / rel
                    if p.is_dir():
                        dst.mkdir(parents=True, exist_ok=True)
                    else:
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(p, dst)
                
                # Copy existing scaler since MLflow might not have it
                shutil.copy2(SCALER_PATH, tmp_dir / "scaler.pkl")

                # Validate
                new_state = _load_validated_state(tmp_dir, tmp_dir / "scaler.pkl")

                # If valid, clear target and copy
                if MODEL_BASE.exists():
                    shutil.rmtree(MODEL_BASE)
                MODEL_BASE.mkdir(parents=True, exist_ok=True)
                
                for p in tmp_dir.rglob("*"):
                    if p.name == "scaler.pkl": continue # Skip the scaler, already at SCALER_PATH
                    rel = p.relative_to(tmp_dir)
                    dst = MODEL_BASE / rel
                    if p.is_dir():
                        dst.mkdir(parents=True, exist_ok=True)
                    else:
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(p, dst)

            # Atomic state switch
            request.app.state.model_state = new_state
            return {"ok": True, "model_uri": used, "bundle": str(MODEL_BASE), 
                    "n_features": len(request.app.state.model_state["features"]), 
                    "threshold": request.app.state.model_state["threshold"]}

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"deploy_registry failed: {e}")

@app.post("/predict")
def predict(req: PredictRequest, request: Request):
    state = request.app.state.model_state
    
    scaler = state["scaler"]
    sess = state["sess"]
    threshold = state["threshold"]
    inp_name = state["inp_name"]
    out_probs_name = state["out_probs_name"]
    features_list = state["features"]

    X = _validate_and_stack(req.instances, features_list)
    Xs = scaler.transform(X).astype(np.float32, copy=False)

    probs2 = sess.run([out_probs_name], {inp_name: Xs})[0]
    p1 = probs2[:, 1].astype(float)

    yhat = (p1 >= threshold).astype(int).tolist()
    return {"probabilities": p1.tolist(), "predictions": yhat, "threshold": threshold}

@app.post("/predict_proba")
def predict_proba(req: PredictProbaRequest, request: Request):
    state = request.app.state.model_state

    scaler = state["scaler"]
    sess = state["sess"]
    inp_name = state["inp_name"]
    out_probs_name = state["out_probs_name"]
    features_list = state["features"]

    X = _validate_and_stack(req.instances, features_list)
    Xs = scaler.transform(X).astype(np.float32, copy=False)

    try:
        probs2 = sess.run([out_probs_name], {inp_name: Xs})[0]
        p1 = probs2[:, 1].astype(float).tolist()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ONNX inference failed: {e}")

    return {"probabilities": p1, "n": len(p1)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("ids_unsw.serve.app:app", host="0.0.0.0", port=8000, reload=False)
