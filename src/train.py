"""Train models and save an evaluation summary and best model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

from src.data import EXPECTED_CLASSES, load_dataset
from src.models import build_models


def train(data_path: str, output_dir: str, label_column: str | None = None,
          test_size: float = 0.2, random_state: int = 42, neighbors: int = 5):
    X, y, feature_columns = load_dataset(data_path, label_column)
    counts = y.value_counts().reindex(EXPECTED_CLASSES, fill_value=0)
    if (counts == 0).any():
        raise ValueError(f"Dataset must contain all four classes. Counts: {counts.to_dict()}")
    if counts.min() < 2:
        raise ValueError("Each class needs at least two samples for a stratified holdout")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    results = {}
    fitted = {}
    for name, model in build_models(random_state, neighbors).items():
        model.fit(X_train, y_train)
        prediction = model.predict(X_test)
        results[name] = {
            "accuracy": float(accuracy_score(y_test, prediction)),
            "macro_f1": float(f1_score(y_test, prediction, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(y_test, prediction, average="weighted", zero_division=0)),
            "classification_report": classification_report(
                y_test, prediction, labels=EXPECTED_CLASSES, output_dict=True, zero_division=0
            ),
            "confusion_matrix": confusion_matrix(y_test, prediction, labels=EXPECTED_CLASSES).tolist(),
        }
        fitted[name] = model
        print(f"{name:24s} accuracy={results[name]['accuracy']:.4f}  macro-F1={results[name]['macro_f1']:.4f}")

    best_name = max(results, key=lambda name: results[name]["macro_f1"])
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model": fitted[best_name],
        "model_name": best_name,
        "feature_columns": feature_columns,
        "classes": list(EXPECTED_CLASSES),
        "image_side": 28,
    }, out / "model.joblib")
    summary = {
        "data_path": str(Path(data_path)),
        "samples": int(len(y)),
        "features": int(X.shape[1]),
        "class_counts": {key: int(value) for key, value in counts.items()},
        "test_size": test_size,
        "random_state": random_state,
        "selected_model": best_name,
        "selection_metric": "macro_f1",
        "models": results,
    }
    (out / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    best_report = results[best_name]["classification_report"]
    pd.DataFrame(best_report).transpose().to_csv(out / "classification_report.csv")
    print(f"\nSelected {best_name} by macro-F1; artifacts saved to {out.resolve()}")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="lbddataset.csv", help="Labeled feature CSV")
    parser.add_argument("--label-column", default=None, help="Target column (auto-detected by default)")
    parser.add_argument("--output-dir", default="artifacts", help="Where model and metrics are saved")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--neighbors", type=int, default=5, help="K for KNN")
    args = parser.parse_args()
    if not 0 < args.test_size < 1:
        parser.error("--test-size must be between 0 and 1")
    if args.neighbors < 1:
        parser.error("--neighbors must be at least 1")
    train(args.data, args.output_dir, args.label_column, args.test_size,
          args.random_state, args.neighbors)


if __name__ == "__main__":
    main()

