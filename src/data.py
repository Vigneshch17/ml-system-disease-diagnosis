"""Dataset loading and image preprocessing helpers."""

from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd
from PIL import Image

EXPECTED_CLASSES = ("NORMAL", "TUBERCULOSIS", "PNEUMONIA", "COVID")
DEFAULT_LABEL_COLUMNS = ("DISEASE", "LABEL", "CLASS", "TARGET")
IMAGE_SIDE = 28


def normalize_label(value: object) -> str:
    """Map common spelling/abbreviation variants to canonical class names."""
    label = re.sub(r"[^A-Z0-9]+", "_", str(value).strip().upper()).strip("_")
    aliases = {
        "TB": "TUBERCULOSIS",
        "TUBERCULOSIS": "TUBERCULOSIS",
        "COVID19": "COVID",
        "COVID_19": "COVID",
        "COVID_19_PNEUMONIA": "COVID",
        "PNEUMONIA": "PNEUMONIA",
        "NORMAL": "NORMAL",
    }
    return aliases.get(label, label)


def load_dataset(path: str | Path, label_column: str | None = None):
    """Load a labeled CSV, returning numeric features and canonical labels."""
    csv_path = Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Dataset CSV not found: {csv_path}")
    frame = pd.read_csv(csv_path)
    if frame.empty:
        raise ValueError(f"Dataset contains no samples: {csv_path}")

    if label_column is None:
        by_upper = {str(column).strip().upper(): column for column in frame.columns}
        label_column = next(
            (by_upper[name] for name in DEFAULT_LABEL_COLUMNS if name in by_upper), None
        )
    elif label_column not in frame.columns:
        raise ValueError(
            f"Label column {label_column!r} not found. Available columns: {list(frame.columns)}"
        )
    if label_column is None:
        raise ValueError(
            "Could not find a label column. Expected one of: "
            + ", ".join(DEFAULT_LABEL_COLUMNS)
        )

    raw_labels = frame.pop(label_column)
    if raw_labels.isna().any():
        raise ValueError("The label column contains missing values")
    y = raw_labels.map(normalize_label)
    non_class = sorted(set(y) - set(EXPECTED_CLASSES))
    if non_class:
        raise ValueError(
            f"Unexpected labels {non_class}; expected classes: {list(EXPECTED_CLASSES)}"
        )
    if y.isna().any():
        raise ValueError("The label column contains missing values")

    # Convert numeric strings while retaining missing values for the train-only imputer.
    X = frame.apply(pd.to_numeric, errors="coerce")
    invalid = frame.notna() & X.isna()
    bad_columns = invalid.columns[invalid.any()].tolist()
    if bad_columns:
        raise ValueError(f"Feature columns contain non-numeric values: {bad_columns[:8]}")
    all_missing = X.columns[X.isna().all()].tolist()
    if all_missing:
        raise ValueError(f"Feature columns contain only missing values: {all_missing[:8]}")
    if X.shape[1] == 0:
        raise ValueError("No feature columns remain after removing the label")
    return X.astype(np.float32), y.astype(str), list(X.columns)


def image_to_features(image: Image.Image) -> np.ndarray:
    """Convert an image to the project's assumed 28×28 grayscale pixel vector."""
    resized = image.convert("L").resize((IMAGE_SIDE, IMAGE_SIDE), Image.Resampling.LANCZOS)
    values = np.asarray(resized, dtype=np.float32).reshape(1, -1)
    return values

