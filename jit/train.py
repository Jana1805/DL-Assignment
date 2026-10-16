"""Train the production defect-prediction model and persist it.

Reproduces notebooks/Deep_MLP (V2).ipynb and exports the result to ONNX.

Usage:
    python -m jit.train --data dataset/apachejit_total.csv --out artifacts
"""

import argparse
import json
import logging
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, average_precision_score, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset

from jit.artifacts import INPUT_NAME, METADATA_FILE, MODEL_FILE, digest
from jit.features import MODEL_FEATURES, RAW_FEATURES, engineer
from jit.model import DROPOUT, HIDDEN_DIMS, DeepMLP, RiskScorer

logger = logging.getLogger(__name__)

LABEL = "buggy"
RANDOM_SEED = 42
TRAIN_YEAR_MAX = 2014
VAL_YEAR_MIN, VAL_YEAR_MAX = 2015, 2016
TEST_YEAR_MIN = 2017
THRESHOLD_GRID = np.arange(0.05, 0.96, 0.05)

LEARNING_RATE = 1e-3
BATCH_SIZE = 256
MAX_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 10


def set_seed(seed: int) -> None:
    """Fix random seeds across all libraries for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_dataset(path: Path) -> pd.DataFrame:
    """Read the ApacheJIT CSV and validate the columns this service needs."""
    frame = pd.read_csv(path, usecols=[*RAW_FEATURES, LABEL, "year"])
    if frame.isnull().any().any():
        raise ValueError("Dataset contains missing values.")
    return frame


def chronological_split(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by commit year so evaluation always uses future commits."""
    train = frame[frame["year"] <= TRAIN_YEAR_MAX]
    val = frame[frame["year"].between(VAL_YEAR_MIN, VAL_YEAR_MAX)]
    test = frame[frame["year"] >= TEST_YEAR_MIN]
    if min(len(train), len(val), len(test)) == 0:
        raise ValueError("Chronological split produced an empty partition.")
    return train, val, test


def select_threshold(y_true: np.ndarray, probs: np.ndarray) -> float:
    """F1-maximising decision threshold, chosen on validation data only."""
    scores = [
        f1_score(y_true, (probs >= t).astype(int), zero_division=0)
        for t in THRESHOLD_GRID
    ]
    return float(round(THRESHOLD_GRID[int(np.argmax(scores))], 2))


def evaluate(y_true: np.ndarray, probs: np.ndarray, threshold: float) -> dict:
    """Test-set metrics, computed as in the MLP notebook."""
    preds = (probs >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, preds),
        "precision": precision_score(y_true, preds, zero_division=0),
        "recall": recall_score(y_true, preds, zero_division=0),
        "f1": f1_score(y_true, preds, zero_division=0),
        "roc_auc": roc_auc_score(y_true, probs),
        "pr_auc": average_precision_score(y_true, probs),
    }


def _loader(X: np.ndarray, y: np.ndarray, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)
    )
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle)


def _run_epoch(model, loader, criterion, optimizer) -> float:
    """Run one epoch. Pass optimizer=None for evaluation."""
    is_train = optimizer is not None
    model.train(mode=is_train)

    total_loss = 0.0
    with torch.set_grad_enabled(is_train):
        for X_batch, y_batch in loader:
            if is_train:
                optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            if is_train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * X_batch.size(0)

    return total_loss / len(loader.dataset)


def train_mlp(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    max_epochs: int = MAX_EPOCHS,
) -> tuple[DeepMLP, int]:
    """Class-weighted training with early stopping on validation loss.

    Inputs are already scaled. Returns the best-epoch model and the number
    of epochs run.
    """
    set_seed(RANDOM_SEED)
    train_loader = _loader(X_train, y_train, shuffle=True)
    val_loader = _loader(X_val, y_val, shuffle=False)

    model = DeepMLP(X_train.shape[1], HIDDEN_DIMS, DROPOUT)
    class_weights = compute_class_weight(
        "balanced", classes=np.array([0, 1]), y=y_train
    )
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    best_state_dict = None

    for epoch in range(1, max_epochs + 1):
        train_loss = _run_epoch(model, train_loader, criterion, optimizer)
        val_loss = _run_epoch(model, val_loader, criterion, None)
        scheduler.step(val_loss)
        logger.info(
            "Epoch %3d | train_loss=%.4f | val_loss=%.4f", epoch, train_loss, val_loss
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            break

    model.load_state_dict(best_state_dict)
    return model, epoch


def predict(scorer: RiskScorer, frame: pd.DataFrame) -> np.ndarray:
    """Probability of the buggy class for each row of raw metrics."""
    features = torch.tensor(engineer(frame).to_numpy(dtype=np.float32))
    with torch.no_grad():
        return scorer(features).numpy()


def export_onnx(scorer: RiskScorer, path: Path) -> None:
    """Write the scaler, MLP and softmax as a single ONNX graph."""
    example = torch.zeros(1, len(MODEL_FEATURES), dtype=torch.float32)
    torch.onnx.export(
        scorer, (example,), str(path),
        input_names=[INPUT_NAME], output_names=["risk_score"],
        dynamic_axes={INPUT_NAME: {0: "batch"}, "risk_score": {0: "batch"}},
        dynamo=False,
    )


def fit(frame: pd.DataFrame, max_epochs: int = MAX_EPOCHS) -> tuple[RiskScorer, dict]:
    """Train on the training years, tune on validation, report on test."""
    train, val, test = chronological_split(frame)
    y_train, y_val, y_test = (
        part[LABEL].astype(int).to_numpy() for part in (train, val, test)
    )

    scaler = StandardScaler().fit(engineer(train))
    mlp, epochs = train_mlp(
        scaler.transform(engineer(train)), y_train,
        scaler.transform(engineer(val)), y_val,
        max_epochs=max_epochs,
    )
    scorer = RiskScorer(mlp, scaler.mean_, scaler.scale_).eval()

    threshold = select_threshold(y_val, predict(scorer, val))
    metrics = evaluate(y_test, predict(scorer, test), threshold)

    summary = {
        "threshold": threshold,
        "epochs": epochs,
        "rows": {"train": len(train), "val": len(val), "test": len(test)},
        "test_metrics": {k: round(float(v), 4) for k, v in metrics.items()},
    }
    return scorer, summary


def fit_and_save(
    frame: pd.DataFrame, out_dir: Path, max_epochs: int = MAX_EPOCHS
) -> dict:
    """Train, export the model and write its metadata."""
    scorer, summary = fit(frame, max_epochs=max_epochs)

    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / MODEL_FILE
    export_onnx(scorer, model_path)

    metadata = {
        "model_version": digest(model_path),
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "torch_version": torch.__version__,
        "features": MODEL_FEATURES,
        **summary,
    }
    (out_dir / METADATA_FILE).write_text(json.dumps(metadata, indent=2))
    return metadata


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("artifacts"))
    args = parser.parse_args()

    metadata = fit_and_save(load_dataset(args.data), args.out)
    logger.info("Saved model %s to %s", metadata["model_version"], args.out)
    logger.info("Threshold: %.2f", metadata["threshold"])
    logger.info("Test metrics: %s", json.dumps(metadata["test_metrics"]))


if __name__ == "__main__":
    main()
