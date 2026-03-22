"""
Unit tests for the production artifact bundle.

These tests run against the committed bundle files and confirm that:
  - the bundle structure is intact
  - metadata follows the canonical schema (schema_version 1.0)
  - no legacy experiment-comparison keys remain in the production bundle
  - the ONNX model input count matches feature_names.json
"""

import json
from pathlib import Path

import pytest

BUNDLE = Path("notebooks/ids_unsw/models/bundle_xgb")

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

EXPECTED_FEATURE_COUNT = 34


# ── Helpers ───────────────────────────────────────────────────────────────────


def _load_features() -> list[str]:
    obj = json.loads((BUNDLE / "feature_names.json").read_text())
    return obj["features"] if isinstance(obj, dict) and "features" in obj else obj


def _load_meta() -> dict:
    return json.loads((BUNDLE / "metadata.json").read_text())


# ── Bundle structure ──────────────────────────────────────────────────────────


def test_bundle_dir_exists():
    assert BUNDLE.exists(), f"Bundle directory not found: {BUNDLE}"


def test_bundle_contains_required_files():
    for fname in ("xgb.onnx", "feature_names.json", "metadata.json"):
        assert (BUNDLE / fname).exists(), f"Missing required file: {BUNDLE / fname}"


# ── Feature names ─────────────────────────────────────────────────────────────


def test_feature_names_is_list():
    assert isinstance(_load_features(), list)


def test_feature_names_count_is_34():
    assert len(_load_features()) == EXPECTED_FEATURE_COUNT


def test_feature_names_are_strings():
    assert all(isinstance(f, str) for f in _load_features())


# ── Metadata schema ───────────────────────────────────────────────────────────


def test_metadata_has_schema_version():
    assert _load_meta().get("schema_version") == "1.0"


def test_metadata_has_all_required_keys():
    meta = _load_meta()
    missing = REQUIRED_METADATA_KEYS - set(meta.keys())
    assert not missing, f"metadata.json missing keys: {missing}"


def test_metadata_has_no_legacy_keys():
    meta = _load_meta()
    found = LEGACY_METADATA_KEYS & set(meta.keys())
    assert not found, (
        f"metadata.json contains legacy keys that must be removed: {found}"
    )


def test_metadata_has_no_nested_xgboost_block():
    meta = _load_meta()
    assert not isinstance(meta.get("xgboost"), dict), (
        "metadata.json contains a legacy nested 'xgboost' comparison block"
    )


def test_metadata_threshold_is_valid_float():
    thr = _load_meta()["threshold"]
    assert isinstance(thr, (int, float))
    assert 0.0 < float(thr) < 1.0, f"threshold {thr!r} is not in (0, 1)"


def test_metadata_n_features_matches_feature_names():
    meta = _load_meta()
    assert int(meta["n_features"]) == len(_load_features())


def test_metrics_at_threshold_keys_are_lowercase():
    metrics = _load_meta()["metrics_at_threshold"]
    upper_keys = [k for k in metrics if k != k.lower()]
    assert not upper_keys, (
        f"metrics_at_threshold has uppercase keys (should be lowercase): {upper_keys}"
    )


# ── ONNX model ────────────────────────────────────────────────────────────────


def test_onnx_is_loadable():
    import onnx
    onnx.load(str(BUNDLE / "xgb.onnx"))


def test_onnx_input_count_matches_feature_names():
    import onnxruntime as ort

    sess = ort.InferenceSession(
        str(BUNDLE / "xgb.onnx"), providers=["CPUExecutionProvider"]
    )
    inp = sess.get_inputs()[0]
    onnx_n = inp.shape[1] if inp.shape and len(inp.shape) > 1 else None
    if onnx_n is not None:
        assert onnx_n == EXPECTED_FEATURE_COUNT, (
            f"ONNX model expects {onnx_n} features, "
            f"but feature_names.json has {EXPECTED_FEATURE_COUNT}"
        )


# ── Full validator smoke ──────────────────────────────────────────────────────


def test_validate_bundle_passes():
    """The canonical validator must report zero errors against the committed bundle."""
    from pathlib import Path

    from ids_unsw.validate_bundle import validate_bundle

    scaler = Path("notebooks/ids_unsw/models/scaler.pkl")
    errors = validate_bundle(BUNDLE, scaler)
    assert errors == [], f"validate_bundle reported errors:\n" + "\n".join(errors)
