"""
Bundle validator for UNSW-NB15_MLOPS.

Verifies that all required bundle files are present and internally consistent:
  - xgb.onnx loads and has the expected input feature count
  - feature_names.json contains a non-empty feature list
  - metadata.json matches the canonical schema (schema_version 1.0)
  - scaler.pkl loads and expects the same feature count

Usage (standalone):
    python -m ids_unsw.validate_bundle

Returns exit code 0 on success, 1 on any failure.
"""

from __future__ import annotations

import json
import os
import pickle
import sys
from pathlib import Path

REQUIRED_METADATA_KEYS = {
    "schema_version",
    "project_name",
    "model_family",
    "artifact_format",
    "n_features",
    "threshold",
    "feature_source",
    "metrics_at_threshold",
    "training_context",
}

LEGACY_METADATA_KEYS = {"rf", "champion", "features"}


def validate_bundle(bundle_dir: Path, scaler_path: Path) -> list[str]:
    """
    Validate the artifact bundle.

    Returns a list of error strings. An empty list means the bundle is valid.
    """
    errors: list[str] = []

    # ── 1. File existence ────────────────────────────────────────────────────
    required = {
        "xgb.onnx": bundle_dir / "xgb.onnx",
        "feature_names.json": bundle_dir / "feature_names.json",
        "metadata.json": bundle_dir / "metadata.json",
    }
    for label, path in required.items():
        if not path.exists():
            errors.append(f"Missing required file: {path}")

    if not scaler_path.exists():
        errors.append(f"Missing scaler: {scaler_path}")

    if errors:
        return errors  # cannot proceed without the files

    # ── 2. Feature names ─────────────────────────────────────────────────────
    feat_obj = json.loads((bundle_dir / "feature_names.json").read_text())
    features: list[str] = (
        feat_obj["features"]
        if isinstance(feat_obj, dict) and "features" in feat_obj
        else feat_obj
    )
    if not isinstance(features, list) or len(features) == 0:
        errors.append("feature_names.json must contain a non-empty list of features.")
        return errors

    n_features = len(features)

    # ── 3. Metadata schema ───────────────────────────────────────────────────
    meta = json.loads((bundle_dir / "metadata.json").read_text())

    for key in REQUIRED_METADATA_KEYS:
        if key not in meta:
            errors.append(f"metadata.json missing required key: '{key}'")

    for key in LEGACY_METADATA_KEYS:
        if key in meta:
            errors.append(
                f"metadata.json contains legacy key '{key}' — remove it "
                f"(model comparison history belongs in MLflow, not the production bundle)"
            )

    # Nested xgboost comparison block (different from model_family string)
    if isinstance(meta.get("xgboost"), dict):
        errors.append(
            "metadata.json contains a legacy nested 'xgboost' comparison block — remove it"
        )

    if "threshold" in meta:
        thr = meta["threshold"]
        if not isinstance(thr, (int, float)) or not (0.0 < float(thr) < 1.0):
            errors.append(
                f"metadata.json 'threshold' must be a float in (0, 1), got: {thr!r}"
            )

    if "n_features" in meta and int(meta["n_features"]) != n_features:
        errors.append(
            f"metadata.json 'n_features' ({meta['n_features']}) does not match "
            f"feature_names.json count ({n_features})"
        )

    # ── 4. ONNX model ────────────────────────────────────────────────────────
    try:
        import onnxruntime as ort

        sess = ort.InferenceSession(
            str(bundle_dir / "xgb.onnx"), providers=["CPUExecutionProvider"]
        )
        inp = sess.get_inputs()[0]
        onnx_n = inp.shape[1] if inp.shape and len(inp.shape) > 1 else None
        if onnx_n is not None and int(onnx_n) != n_features:
            errors.append(
                f"ONNX model expects {onnx_n} input features, "
                f"but feature_names.json has {n_features}"
            )
    except ImportError:
        errors.append("onnxruntime is not installed — cannot validate ONNX model")
    except Exception as exc:
        errors.append(f"Failed to load ONNX model: {exc}")

    # ── 5. Scaler ─────────────────────────────────────────────────────────────
    try:
        with open(scaler_path, "rb") as fh:
            scaler = pickle.load(fh)
        if hasattr(scaler, "n_features_in_"):
            scaler_n = int(scaler.n_features_in_)
            if scaler_n != n_features:
                errors.append(
                    f"Scaler expects {scaler_n} features, "
                    f"but feature_names.json has {n_features}"
                )
    except Exception as exc:
        errors.append(f"Failed to load scaler: {exc}")

    return errors


def main() -> None:
    bundle_dir = Path(
        os.getenv("IDS_BUNDLE_DIR", "notebooks/ids_unsw/models/bundle_xgb")
    )
    scaler_path = Path(
        os.getenv("IDS_SCALER_PATH", "notebooks/ids_unsw/models/scaler.pkl")
    )

    print(f"Validating bundle : {bundle_dir}")
    print(f"Scaler            : {scaler_path}")

    errors = validate_bundle(bundle_dir, scaler_path)

    if errors:
        print("\nBundle validation FAILED:")
        for err in errors:
            print(f"  FAIL  {err}")
        sys.exit(1)

    print("Bundle validation PASSED — all checks OK.")
    sys.exit(0)


if __name__ == "__main__":
    main()
