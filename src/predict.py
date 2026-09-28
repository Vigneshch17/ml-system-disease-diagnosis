"""Load a trained classifier and predict labels for flattened features."""

from pathlib import Path

import joblib
import numpy as np


def load_bundle(path: str | Path = "artifacts/model.joblib"):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Model not found at {path}. Train it first with python -m src.train.")
    return joblib.load(path)


def predict_features(features: np.ndarray, bundle: dict):
    values = np.asarray(features, dtype=np.float32)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    expected = len(bundle["feature_columns"])
    if values.shape[1] != expected:
        raise ValueError(f"Expected {expected} features, got {values.shape[1]}")
    model = bundle["model"]
    labels = model.predict(values)
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(values)
        classes = model.named_steps["classifier"].classes_
        return [(str(label), {str(c): float(p) for c, p in zip(classes, row)})
                for label, row in zip(labels, probabilities)]
    return [(str(label), {}) for label in labels]

