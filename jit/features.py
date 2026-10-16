"""Feature engineering shared by training and serving (the v2 pipeline).

Keeping this in one module guarantees that the model sees identically
transformed inputs at training time and at inference time.
"""

import numpy as np
import pandas as pd

RAW_FEATURES = [
    "la", "ld", "nf", "nd", "ns", "ent",
    "ndev", "age", "nuc", "aexp", "arexp", "asexp",
]
DERIVED_FEATURES = [
    "total_churn", "churn_per_file", "files_per_dir",
    "add_delete_ratio", "exp_ratio",
]
MODEL_FEATURES = RAW_FEATURES + DERIVED_FEATURES

# Heavily right-skewed columns (skewness > 1) receive log1p. `ent` is
# mildly skewed and is left as is, matching the v2 preprocessing notebook.
LOG_COLUMNS = [
    "la", "ld", "nf", "nd", "ns", "ndev", "age", "nuc",
    "aexp", "arexp", "asexp", "total_churn", "churn_per_file",
]


def engineer(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the 17 model features, in a fixed column order.

    Derived ratios are computed on raw values first; the log transform is
    applied afterwards, as in the v2 preprocessing notebook.
    """
    missing = [c for c in RAW_FEATURES if c not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out = frame[RAW_FEATURES].astype("float64").copy()
    if (out < 0).any().any():
        raise ValueError("Commit metrics must be non-negative.")

    out["total_churn"] = out["la"] + out["ld"]
    out["churn_per_file"] = out["total_churn"] / out["nf"].replace(0, 1)
    out["files_per_dir"] = out["nf"] / out["nd"].replace(0, 1)
    out["add_delete_ratio"] = out["la"] / (out["ld"] + 1)
    out["exp_ratio"] = out["arexp"] / (out["aexp"] + 1)

    out[LOG_COLUMNS] = np.log1p(out[LOG_COLUMNS])
    return out[MODEL_FEATURES]
