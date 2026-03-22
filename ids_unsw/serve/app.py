from __future__ import annotations

import os, json, pickle, shutil, glob, tempfile, pathlib
from pathlib import Path
from typing import List, Dict, Any, Optional
from threading import Lock

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException, Body, Depends, Security, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
import mlflow
from mlflow import tracking
from mlflow.tracking import MlflowClient
from mlflow.artifacts import download_artifacts

# --- add near the top with other imports ---
# Base/bundle paths (env overrideable)
MODEL_BASE = Path(os.getenv("IDS_BUNDLE_DIR", "notebooks/ids_unsw/models/bundle_xgb"))
SCALER_PATH = Path(os.getenv("IDS_SCALER_PATH", "notebooks/ids_unsw/models/scaler.pkl"))

# This is derived from the context, assuming MODELS is the parent of BUNDLE
MODELS = MODEL_BASE.parent

# ---- Auth ----
API_TOKEN = os.getenv("IDS_API_TOKEN")
if not API_TOKEN:
    raise RuntimeError("IDS_API_TOKEN environment variable is required.")

# ---- Docs exposure ----
# Set IDS_EXPOSE_DOCS=true to enable /docs and /openapi.json (e.g. during local dev).
# Defaults to disabled.
_expose_docs = os.getenv("IDS_EXPOSE_DOCS", "false").lower() == "true"
_docs_url    = "/docs"        if _expose_docs else None
_openapi_url = "/openapi.json" if _expose_docs else None

_bearer = HTTPBearer(auto_error=False)
def require_token(request: Request, creds: HTTPAuthorizationCredentials = Security(_bearer)):
    if request.url.path == "/health":
        return
    if not creds or creds.scheme.lower() != "bearer" or creds.credentials != API_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")

_state_lock = Lock()

def _load_bundle():
    """Loads all model artifacts from disk."""
    # accept xgb.onnx or model.onnx
    onnx_path = MODEL_BASE / "xgb.onnx"
    if not onnx_path.exists():
        alt = MODEL_BASE / "model.onnx"
        if alt.exists():
            onnx_path = alt
        else:
            raise RuntimeError(f"Missing ONNX model at {onnx_path} or {alt}")

    if not (MODEL_BASE / "feature_names.json").exists():
        raise RuntimeError(f"Missing features file at {MODEL_BASE/'feature_names.json'}")
    if not (MODEL_BASE / "metadata.json").exists():
        raise RuntimeError(f"Missing metadata at {MODEL_BASE/'metadata.json'}")
    if not SCALER_PATH.exists():
        raise RuntimeError(f"Missing scaler at {SCALER_PATH}")

    # features can be a list or {"features": [...]}
    feat_obj = json.loads((MODEL_BASE / "feature_names.json").read_text())
    features = feat_obj["features"] if isinstance(feat_obj, dict) and "features" in feat_obj else feat_obj

    meta = json.loads((MODEL_BASE / "metadata.json").read_text())
    thr = float(meta["threshold"])

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inp_name = sess.get_inputs()[0].name
    out_probs_name = sess.get_outputs()[1].name

    with open(SCALER_PATH, "rb") as f:
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

# Initialize once on startup
try:
    STATE  # reuse if already defined
except NameError:
    STATE = _load_bundle()

# ------------ FastAPI setup ------------
app = FastAPI(
    title="UNSW-NB15_MLOPS — IDS API",
    version="1.0",
    docs_url=_docs_url,
    openapi_url=_openapi_url,
    dependencies=[Depends(require_token)],
)

@app.on_event("startup")
async def startup_validation():
    """Validate bundle integrity at startup using the canonical validator."""
    from ids_unsw.validate_bundle import validate_bundle

    errors = validate_bundle(MODEL_BASE, SCALER_PATH)
    if errors:
        msg = "Bundle validation failed at startup:\n" + "\n".join(
            f"  - {e}" for e in errors
        )
        raise RuntimeError(msg)


class PredictRequest(BaseModel):
    # A list of feature dictionaries (keys must match feature_names.json)
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
    import json
    # Note: Using MODELS and MODEL_BASE which correspond to the user's
    # intent for MODEL_DIR and BUNDLE_DIR
    targets = [
        MODELS / "metadata.json",
        MODEL_BASE / "metadata.json",
    ]
    for p in targets:
        if not p.parent.exists(): continue
        try:
            meta = json.loads(p.read_text())
        except Exception:
            meta = {}
        meta["threshold"] = float(thr)
        p.write_text(json.dumps(meta, indent=2))


def _validate_and_stack(rows: List[Dict[str, Any]]) -> np.ndarray:
    """Ensure each row has exactly the expected features and order them."""
    with _state_lock:
        features = STATE["features"]
    
    fset = set(features)
    X_list = []
    for i, row in enumerate(rows):
        keys = set(row.keys())
        missing = list(fset - keys)
        extra = list(keys - fset)
        if missing or extra:
            raise HTTPException(
                status_code=400,
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

def _copy_if_exists(src_dir: str, dst_dir: pathlib.Path, filenames=("feature_names.json", "metadata.json")):
    for name in filenames:
        p = pathlib.Path(src_dir) / name
        if p.exists():
            shutil.copy2(p, dst_dir / name)

@app.get("/health")
def health():
    with _state_lock:
        return {
            "status": "ok",
            "model_bundle_dir": str(MODEL_BASE),
            "scaler_path": str(SCALER_PATH),
            "n_features": len(STATE["features"]),
            "threshold": STATE["threshold"],
        }

@app.get("/features")
def features():
    with _state_lock:
        return {"features": STATE["features"]}

@app.get("/metadata")
def metadata():
    with _state_lock:
        return STATE["meta"]

# --- add these endpoints somewhere after FastAPI app is created ---
@app.get("/meta")
def meta():
    with _state_lock:
        return {
            "model": "xgboost_onnx",
            "n_features": len(STATE["features"]),
            "threshold": STATE["threshold"],
            "meta": STATE["meta"],
        }

@app.post("/reload")
def reload_model():
    global STATE
    with _state_lock:
        try:
            STATE = _load_bundle()
            return {
                "ok": True,
                "n_features": len(STATE["features"]),
                "threshold": STATE["threshold"],
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to reload model: {e}")

@app.post("/set_threshold")
def set_threshold(inp: ThresholdIn):
    """
    Set decision threshold in metadata.json (both places), then hot-reload.
    Returns the live threshold the server is using.
    """
    thr = float(inp.threshold)
    _write_threshold_files(thr)

    # re-use your existing reload logic
    reload_info = reload_model()

    # return current live state from the reloaded STATE dictionary
    with _state_lock:
        live_threshold = STATE["threshold"]
        n_features = len(STATE["features"])

    return {
        "ok": True,
        "n_features": n_features,
        "threshold": live_threshold,
        "info": reload_info,
    }

@app.post("/deploy_registry")
def deploy_registry(req: DeployReq):
    """
    Download a registered ONNX model from MLflow and hot-reload.
    Try, in order:
      1) models:/<name>/<version>
      2) mv.source (often models:/m-xxxx or runs:/.../xgb)
      3) runs:/<mv.run_id>/xgb   (explicit fallback to run artifacts)
    """
    mlflow.set_tracking_uri(req.tracking_uri)
    client = MlflowClient()

    tried = []
    def _try(uri: str):
        tried.append(uri)
        return download_artifacts(uri)

    try:
        # fetch model version metadata once
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

        # Copy into our bundle dir and reload
        local_dir = Path(local)
        onnx_files = list(local_dir.rglob("*.onnx"))
        if not onnx_files:
            raise HTTPException(status_code=400, detail=f"No .onnx found under downloaded path: {local_dir}")

        MODEL_BASE.mkdir(parents=True, exist_ok=True)
        for p in local_dir.rglob("*"):
            rel = p.relative_to(local_dir)
            dst = MODEL_BASE / rel
            if p.is_dir():
                dst.mkdir(parents=True, exist_ok=True)
            else:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, dst)

        info = reload_model()
        return {"ok": True, "model_uri": used, "bundle": str(MODEL_BASE), **info}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"deploy_registry failed: {e}")


@app.post("/predict")
def predict(req: PredictRequest):
    # Get a consistent snapshot of the state under the lock
    with _state_lock:
        scaler = STATE["scaler"]
        sess = STATE["sess"]
        threshold = STATE["threshold"]
        inp_name = STATE["inp_name"]
        out_probs_name = STATE["out_probs_name"]

    # 1) validate + order (can happen outside lock)
    X = _validate_and_stack(req.instances)  # shape (n, n_features)

    # 2) scale exactly like training
    Xs = scaler.transform(X).astype(np.float32, copy=False)

    # 3) ONNX inference -> probabilities for class 1
    probs2 = sess.run([out_probs_name], {inp_name: Xs})[0]  # (n, 2)
    p1 = probs2[:, 1].astype(float)

    # 4) threshold to labels
    yhat = (p1 >= threshold).astype(int).tolist()
    return {"probabilities": p1.tolist(), "predictions": yhat, "threshold": threshold}

@app.post("/predict_proba")
def predict_proba(req: PredictProbaRequest):
    # Get a consistent snapshot of the state under the lock
    with _state_lock:
        scaler = STATE["scaler"]
        sess = STATE["sess"]
        inp_name = STATE["inp_name"]
        out_probs_name = STATE["out_probs_name"]

    # 1) validate + order (can happen outside lock)
    X = _validate_and_stack(req.instances)

    # 2) scale like training
    Xs = scaler.transform(X).astype(np.float32, copy=False)

    # 3) run ONNX to get probabilities
    try:
        # most XGBoost-ONNX exports return probs in the second output [1]
        probs2 = sess.run([out_probs_name], {inp_name: Xs})[0]  # (n, 2)
        p1 = probs2[:, 1].astype(float).tolist()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ONNX inference failed: {e}")

    return {"probabilities": p1, "n": len(p1)}

# Optional: uvicorn entrypoint (so you can: python -m ids_unsw.serve.app)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("ids_unsw.serve.app:app", host="0.0.0.0", port=8000, reload=False)

