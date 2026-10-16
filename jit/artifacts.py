"""Locating, verifying and loading persisted model artifacts.

Artifacts are build outputs produced by `jit.train`: an ONNX graph and a
metadata file that records its content hash.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd

from jit.features import MODEL_FEATURES, engineer

MODEL_FILE = "model.onnx"
METADATA_FILE = "metadata.json"
INPUT_NAME = "features"


def digest(path: Path) -> str:
    """Short content hash used as the model version."""
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


class Scorer:
    """Runs the exported graph on raw commit metrics."""

    def __init__(self, model_path: Path):
        self._session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )

    def score(self, frame: pd.DataFrame) -> np.ndarray:
        """Probability of the buggy class for each row of raw metrics."""
        features = engineer(frame).to_numpy(dtype=np.float32)
        return self._session.run(None, {INPUT_NAME: features})[0]


def load_artifacts(model_dir: Path) -> tuple[Scorer, dict]:
    """Load the model and its metadata, failing fast on any mismatch."""
    model_path = model_dir / MODEL_FILE
    metadata_path = model_dir / METADATA_FILE

    if not model_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(
            f"Expected {MODEL_FILE} and {METADATA_FILE} in '{model_dir}'. "
            "Run `python -m jit.train` first."
        )

    metadata = json.loads(metadata_path.read_text())

    if metadata["features"] != MODEL_FEATURES:
        raise RuntimeError(
            "Model was trained on a different feature list than this code "
            "produces. Retrain with this version of jit.features."
        )
    if digest(model_path) != metadata["model_version"]:
        raise RuntimeError(f"{MODEL_FILE} does not match {METADATA_FILE}.")

    return Scorer(model_path), metadata
