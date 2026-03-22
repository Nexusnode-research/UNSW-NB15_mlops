"""
API integration tests for UNSW-NB15_MLOPS.

Uses FastAPI's TestClient (in-process — no running server required).
These tests prove the core API contract:
  - /health is public
  - scoring/admin endpoints require Bearer auth
  - predictions return the expected shape
  - threshold changes persist within the session
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

# Set required env vars before the app module is imported.
os.environ.setdefault("IDS_API_TOKEN", "ci-test-token-abc123")
os.environ.setdefault("IDS_BUNDLE_DIR", "notebooks/ids_unsw/models/bundle_xgb")
os.environ.setdefault("IDS_SCALER_PATH", "notebooks/ids_unsw/models/scaler.pkl")
os.environ.setdefault("IDS_EXPOSE_DOCS", "false")

from fastapi.testclient import TestClient  # noqa: E402
from ids_unsw.serve.app import app  # noqa: E402

TOKEN = os.environ["IDS_API_TOKEN"]
AUTH = {"Authorization": f"Bearer {TOKEN}"}
WRONG_AUTH = {"Authorization": "Bearer definitely-wrong"}


@pytest.fixture(scope="module")
def client():
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


# ── /health — public endpoint ─────────────────────────────────────────────────


def test_health_returns_200_without_auth(client):
    r = client.get("/health")
    assert r.status_code == 200


def test_health_body_contains_status_ok(client):
    r = client.get("/health")
    assert r.json()["status"] == "ok"


def test_health_returns_200_with_wrong_token(client):
    """Wrong token must NOT block /health — it is unconditionally public."""
    r = client.get("/health", headers=WRONG_AUTH)
    assert r.status_code == 200


def test_health_exposes_threshold(client):
    r = client.get("/health")
    body = r.json()
    assert "threshold" in body
    assert isinstance(body["threshold"], float)


# ── Auth enforcement ──────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "method,path,body",
    [
        ("GET",  "/features",      None),
        ("GET",  "/metadata",      None),
        ("POST", "/predict",       {"instances": []}),
        ("POST", "/predict_proba", {"instances": []}),
        ("POST", "/reload",        None),
    ],
)
def test_endpoint_requires_auth_no_header(client, method, path, body):
    r = client.request(method, path, json=body)
    assert r.status_code == 401, (
        f"{method} {path} should return 401 without auth, got {r.status_code}"
    )


@pytest.mark.parametrize(
    "method,path,body",
    [
        ("GET",  "/features",      None),
        ("GET",  "/metadata",      None),
        ("POST", "/predict",       {"instances": []}),
        ("POST", "/predict_proba", {"instances": []}),
    ],
)
def test_endpoint_requires_auth_wrong_token(client, method, path, body):
    r = client.request(method, path, json=body, headers=WRONG_AUTH)
    assert r.status_code == 401, (
        f"{method} {path} should return 401 with wrong token, got {r.status_code}"
    )


# ── /features ─────────────────────────────────────────────────────────────────


def test_features_returns_34_names(client):
    r = client.get("/features", headers=AUTH)
    assert r.status_code == 200
    feats = r.json()["features"]
    assert isinstance(feats, list)
    assert len(feats) == 34


def test_features_are_strings(client):
    r = client.get("/features", headers=AUTH)
    assert all(isinstance(f, str) for f in r.json()["features"])


# ── /metadata ─────────────────────────────────────────────────────────────────


def test_metadata_has_schema_version(client):
    r = client.get("/metadata", headers=AUTH)
    assert r.status_code == 200
    assert r.json().get("schema_version") == "1.0"


# ── /predict ─────────────────────────────────────────────────────────────────


def _zero_payload(client) -> dict:
    r = client.get("/features", headers=AUTH)
    features = r.json()["features"]
    return {"instances": [{f: 0.0 for f in features}]}


def test_predict_valid_payload_returns_200(client):
    r = client.post("/predict", json=_zero_payload(client), headers=AUTH)
    assert r.status_code == 200


def test_predict_response_shape(client):
    r = client.post("/predict", json=_zero_payload(client), headers=AUTH)
    body = r.json()
    assert "predictions" in body
    assert "probabilities" in body
    assert "threshold" in body
    assert len(body["predictions"]) == 1
    assert len(body["probabilities"]) == 1


def test_predict_predictions_are_binary(client):
    r = client.post("/predict", json=_zero_payload(client), headers=AUTH)
    for p in r.json()["predictions"]:
        assert p in (0, 1)


def test_predict_invalid_features_returns_error(client):
    payload = {"instances": [{"nonexistent_feature": 1.0}]}
    r = client.post("/predict", json=payload, headers=AUTH)
    assert r.status_code in (400, 422)


def test_predict_missing_instances_key_returns_422(client):
    r = client.post("/predict", json={"wrong_key": []}, headers=AUTH)
    assert r.status_code == 422


# ── /predict_proba ────────────────────────────────────────────────────────────


def test_predict_proba_valid_payload_returns_200(client):
    r = client.post("/predict_proba", json=_zero_payload(client), headers=AUTH)
    assert r.status_code == 200


def test_predict_proba_returns_probabilities_in_0_1(client):
    r = client.post("/predict_proba", json=_zero_payload(client), headers=AUTH)
    for p in r.json()["probabilities"]:
        assert 0.0 <= p <= 1.0, f"Probability {p} is outside [0, 1]"


# ── /set_threshold ────────────────────────────────────────────────────────────


def test_set_threshold_requires_auth(client):
    r = client.post("/set_threshold", json={"threshold": 0.5})
    assert r.status_code == 401


def test_set_threshold_persists_to_health(client):
    original = client.get("/health").json()["threshold"]
    new_thr = 0.42

    r = client.post("/set_threshold", json={"threshold": new_thr}, headers=AUTH)
    assert r.status_code == 200
    assert abs(r.json()["threshold"] - new_thr) < 1e-6

    health_thr = client.get("/health").json()["threshold"]
    assert abs(health_thr - new_thr) < 1e-6

    # Restore original threshold so test ordering doesn't matter
    client.post("/set_threshold", json={"threshold": original}, headers=AUTH)


def test_set_threshold_rejects_out_of_range(client):
    r = client.post("/set_threshold", json={"threshold": 1.5}, headers=AUTH)
    assert r.status_code == 422


# ── /docs controlled by IDS_EXPOSE_DOCS ──────────────────────────────────────


def test_docs_not_exposed_when_env_false(client):
    """IDS_EXPOSE_DOCS=false (default) — /docs must return 404."""
    r = client.get("/docs")
    assert r.status_code == 404


def test_openapi_not_exposed_when_env_false(client):
    r = client.get("/openapi.json")
    assert r.status_code == 404


# ── Hot Reload Safety  ──────────────────────────────────────

def test_reload_fails_safely_and_keeps_original_state(client, tmp_path):
    """Proves that a broken bundle on disk rejects hot reload and keeps the API alive."""
    bad_bundle = tmp_path / "bad_bundle"
    bad_bundle.mkdir()
    (bad_bundle / "metadata.json").write_text("{}") # Invalid schema missing threshold
    
    # Temporarily redirect the bundle path using the global module
    import ids_unsw.serve.app as serving_module
    old_base = serving_module.MODEL_BASE
    serving_module.MODEL_BASE = bad_bundle
    
    r = client.post("/reload", headers=AUTH)
    assert r.status_code == 500
    assert "validation failed" in r.json()["detail"].lower() or "missing schema" in r.json()["detail"].lower() or "missing features" in r.json()["detail"].lower()
    
    # Revert redirect
    serving_module.MODEL_BASE = old_base
    
    # Confirm old state still works perfectly
    r2 = client.get("/health")
    assert r2.status_code == 200
    assert r2.json()["status"] == "ok"
