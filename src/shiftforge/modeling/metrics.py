"""Deterministic regression metrics for common evaluation samples."""

from __future__ import annotations

import numpy as np


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    truth = np.asarray(y_true, dtype=np.float64).ravel()
    pred = np.asarray(y_pred, dtype=np.float64).ravel()
    if truth.shape != pred.shape or not len(truth):
        raise ValueError("targets and predictions must be nonempty and equally sized")
    if not (np.isfinite(truth).all() and np.isfinite(pred).all()):
        raise ValueError("targets and predictions must be finite")
    error = np.abs(truth - pred)
    return {
        "mae_minutes": float(error.mean()),
        "rmse_minutes": float(np.sqrt(np.mean(np.square(truth - pred)))),
        "p90_absolute_error_minutes": float(np.quantile(error, 0.90)),
        "negative_predictions": int(np.count_nonzero(pred < 0.0)),
    }


def select_model(results: dict[str, dict[str, float | int]]) -> str:
    if not results:
        raise ValueError("no fitted models to compare")
    return min(results, key=lambda name: (float(results[name]["mae_minutes"]), name))


def training_indices(n_rows: int, maximum: int, seed: int) -> np.ndarray:
    """A seeded subsample of the training partition, never of validation."""
    if n_rows < 1 or maximum < 1:
        raise ValueError("training row counts must be positive")
    if n_rows <= maximum:
        return np.arange(n_rows, dtype=np.int64)
    return np.sort(np.random.default_rng(seed).choice(n_rows, size=maximum, replace=False))
