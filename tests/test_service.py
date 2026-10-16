import json
import shutil

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from jit.artifacts import Scorer, digest
from jit.features import MODEL_FEATURES, RAW_FEATURES, engineer
from jit.train import (
    THRESHOLD_GRID, TRAIN_YEAR_MAX, VAL_YEAR_MAX, VAL_YEAR_MIN, TEST_YEAR_MIN,
    chronological_split, export_onnx, fit, fit_and_save, predict, select_threshold,
)

TEST_EPOCHS = 15

SAFE_COMMIT = dict(
    la=5, ld=2, nf=1, nd=1, ns=1, ent=0.2,
    ndev=2, age=30, nuc=5, aexp=1500, arexp=300, asexp=800,
)
RISKY_COMMIT = dict(
    la=3000, ld=500, nf=25, nd=8, ns=4, ent=3.5,
    ndev=15, age=5, nuc=40, aexp=3, arexp=3, asexp=0,
)


def make_synthetic_frame(n: int = 8000, seed: int = 0) -> pd.DataFrame:
    """Synthetic commits where larger changes by newer authors are riskier."""
    rng = np.random.default_rng(seed)
    la = rng.lognormal(3.0, 1.2, n).astype(int)
    ld = rng.lognormal(2.5, 1.2, n).astype(int)
    nf = rng.integers(1, 30, n)
    nd = np.minimum(nf, rng.integers(1, 10, n))
    ns = np.minimum(nd, rng.integers(1, 5, n))
    aexp = rng.integers(0, 2000, n).astype(float)

    logit = (
        0.6 * (np.log1p(la) - 3.5)
        + 0.03 * (nf - 15)
        - 0.5 * (np.log1p(aexp) - 6)
        - 1.0
    )
    buggy = rng.random(n) < 1 / (1 + np.exp(-logit))

    return pd.DataFrame({
        "la": la, "ld": ld, "nf": nf, "nd": nd, "ns": ns,
        "ent": rng.uniform(0, 4, n),
        "ndev": rng.uniform(0, 20, n),
        "age": rng.exponential(50, n),
        "nuc": rng.exponential(20, n),
        "aexp": aexp,
        "arexp": rng.uniform(0, 1, n) * aexp,
        "asexp": rng.uniform(0, 1, n) * aexp,
        "year": rng.integers(2003, 2020, n),
        "buggy": buggy,
    })


@pytest.fixture(scope="session")
def synthetic_frame() -> pd.DataFrame:
    return make_synthetic_frame()


@pytest.fixture(scope="session")
def artifact_dir(tmp_path_factory, synthetic_frame):
    out_dir = tmp_path_factory.mktemp("artifacts")
    fit_and_save(synthetic_frame, out_dir, max_epochs=TEST_EPOCHS)
    return out_dir


@pytest.fixture
def client(artifact_dir, monkeypatch):
    monkeypatch.setenv("MODEL_DIR", str(artifact_dir))
    with TestClient(create_app()) as test_client:
        yield test_client


# ---------------------------------------------------------------- features

def test_engineer_matches_v2_definitions():
    row = dict(la=100, ld=20, nf=4, nd=2, ns=1, ent=1.5,
               ndev=3, age=10, nuc=5, aexp=50, arexp=10, asexp=7)
    out = engineer(pd.DataFrame([row])).iloc[0]

    assert list(out.index) == MODEL_FEATURES
    assert out["la"] == pytest.approx(np.log1p(100))
    assert out["ent"] == pytest.approx(1.5)  # not log-transformed
    assert out["total_churn"] == pytest.approx(np.log1p(120))
    assert out["churn_per_file"] == pytest.approx(np.log1p(30))
    assert out["files_per_dir"] == pytest.approx(2.0)  # not log-transformed
    assert out["add_delete_ratio"] == pytest.approx(100 / 21)
    assert out["exp_ratio"] == pytest.approx(10 / 51)


def test_engineer_rejects_missing_columns():
    with pytest.raises(ValueError, match="Missing required columns"):
        engineer(pd.DataFrame([{"la": 1}]))


def test_engineer_rejects_negative_values():
    with pytest.raises(ValueError, match="non-negative"):
        engineer(pd.DataFrame([{**SAFE_COMMIT, "la": -1}]))


# ---------------------------------------------------------------- training

def test_chronological_split_respects_boundaries(synthetic_frame):
    train, val, test = chronological_split(synthetic_frame)

    assert train["year"].max() <= TRAIN_YEAR_MAX
    assert val["year"].between(VAL_YEAR_MIN, VAL_YEAR_MAX).all()
    assert test["year"].min() >= TEST_YEAR_MIN
    assert len(train) + len(val) + len(test) == len(synthetic_frame)
    assert not (set(train.index) & set(val.index) | set(val.index) & set(test.index))


def test_chronological_split_rejects_empty_partition(synthetic_frame):
    with pytest.raises(ValueError, match="empty partition"):
        chronological_split(synthetic_frame[synthetic_frame["year"] < 2010])


def test_select_threshold_stays_on_grid():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 500)
    threshold = select_threshold(y, rng.random(500))
    assert THRESHOLD_GRID.min() <= threshold <= THRESHOLD_GRID.max() + 1e-9


def test_metadata_matches_artifacts(artifact_dir):
    metadata = json.loads((artifact_dir / "metadata.json").read_text())

    assert metadata["features"] == MODEL_FEATURES
    assert metadata["model_version"] == digest(artifact_dir / "model.onnx")
    assert 0.05 <= metadata["threshold"] <= 0.95
    assert metadata["test_metrics"]["roc_auc"] > 0.6  # learned signal on synthetic data


def test_onnx_export_matches_pytorch(synthetic_frame, tmp_path):
    scorer, _ = fit(synthetic_frame, max_epochs=2)
    export_onnx(scorer, tmp_path / "model.onnx")
    rows = synthetic_frame[RAW_FEATURES].head(200)

    exported = Scorer(tmp_path / "model.onnx").score(rows)

    assert exported.shape == (200,)
    assert exported == pytest.approx(predict(scorer, rows), abs=1e-5)


# ---------------------------------------------------------------- API

def test_health_reports_model_version(client, artifact_dir):
    body = client.get("/health").json()
    metadata = json.loads((artifact_dir / "metadata.json").read_text())
    assert body == {"status": "ok", "model_version": metadata["model_version"]}


def test_predict_returns_valid_response(client):
    response = client.post("/predict", json=SAFE_COMMIT)

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"risk_score", "is_risky", "threshold", "model_version"}
    assert 0.0 <= body["risk_score"] <= 1.0


def test_risky_commit_scores_higher_than_safe_commit(client):
    safe = client.post("/predict", json=SAFE_COMMIT).json()
    risky = client.post("/predict", json=RISKY_COMMIT).json()

    assert risky["risk_score"] > safe["risk_score"]
    assert risky["is_risky"] is True
    assert safe["is_risky"] is False


@pytest.mark.parametrize("override", [
    {"la": -1}, {"nf": 0}, {"nd": 0}, {"ent": -0.1}, {"aexp": -5},
    {"la": "many"}, {"unknown_field": 1},
])
def test_predict_rejects_invalid_input(client, override):
    response = client.post("/predict", json={**SAFE_COMMIT, **override})
    assert response.status_code == 422


def test_predict_rejects_missing_field(client):
    payload = {k: v for k, v in SAFE_COMMIT.items() if k != "aexp"}
    assert client.post("/predict", json=payload).status_code == 422


def test_unexpected_error_returns_500_without_leaking_details(artifact_dir, monkeypatch):
    class Broken:
        def score(self, frame):
            raise RuntimeError("secret internal detail")

    monkeypatch.setenv("MODEL_DIR", str(artifact_dir))
    with TestClient(create_app(), raise_server_exceptions=False) as test_client:
        test_client.app.state.model = Broken()
        response = test_client.post("/predict", json=SAFE_COMMIT)

    assert response.status_code == 500
    assert response.json() == {"detail": "Internal server error"}


def test_predict_requires_api_key_when_configured(client, monkeypatch):
    monkeypatch.setenv("API_KEY", "s3cret")

    assert client.post("/predict", json=SAFE_COMMIT).status_code == 401
    wrong = client.post("/predict", json=SAFE_COMMIT, headers={"X-API-Key": "nope"})
    assert wrong.status_code == 401
    right = client.post("/predict", json=SAFE_COMMIT, headers={"X-API-Key": "s3cret"})
    assert right.status_code == 200
    assert client.get("/health").status_code == 200  # health stays open


# ------------------------------------------------------- startup fail-fast

def test_startup_fails_when_artifacts_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_DIR", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="jit.train"):
        with TestClient(create_app()):
            pass


def test_startup_fails_on_feature_list_mismatch(artifact_dir, tmp_path, monkeypatch):
    broken = tmp_path / "artifacts"
    shutil.copytree(artifact_dir, broken)
    metadata = json.loads((broken / "metadata.json").read_text())
    metadata["features"] = RAW_FEATURES
    (broken / "metadata.json").write_text(json.dumps(metadata))

    monkeypatch.setenv("MODEL_DIR", str(broken))
    with pytest.raises(RuntimeError, match="Retrain"):
        with TestClient(create_app()):
            pass


def test_startup_fails_when_model_file_is_modified(artifact_dir, tmp_path, monkeypatch):
    broken = tmp_path / "artifacts"
    shutil.copytree(artifact_dir, broken)
    with open(broken / "model.onnx", "ab") as f:
        f.write(b"tampered")

    monkeypatch.setenv("MODEL_DIR", str(broken))
    with pytest.raises(RuntimeError, match="does not match"):
        with TestClient(create_app()):
            pass
