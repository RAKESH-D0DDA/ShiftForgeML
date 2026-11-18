"""Metrics, deterministic sampling, model selection and estimator behavior."""

import numpy as np
import pytest

from shiftforge.modeling.estimators import FEATURES, build_estimators
from shiftforge.modeling.metrics import regression_metrics, select_model, training_indices


def test_metrics_exact_values():
    metrics = regression_metrics(np.array([1, 3, 5]), np.array([2, 1, 6]))
    assert metrics["mae_minutes"] == pytest.approx(4 / 3)
    assert metrics["rmse_minutes"] == pytest.approx(np.sqrt(2))
    assert metrics["p90_absolute_error_minutes"] == pytest.approx(1.8)
    assert metrics["negative_predictions"] == 0


def test_invalid_prediction_vectors_rejected():
    with pytest.raises(ValueError, match="finite"):
        regression_metrics(np.array([1.0]), np.array([float("nan")]))
    with pytest.raises(ValueError, match="nonempty"):
        regression_metrics(np.array([1.0]), np.array([1.0, 2.0]))


def test_training_sampling_is_repeatable_and_seeded():
    first = training_indices(1000, 120, 42)
    second = training_indices(1000, 120, 42)
    third = training_indices(1000, 120, 17)
    assert np.array_equal(first, second)
    assert not np.array_equal(first, third)
    assert len(set(first)) == 120
    assert np.all(np.diff(first) > 0)


def test_sampling_leaves_small_datasets_unchanged():
    assert np.array_equal(training_indices(4, 10, 42), np.arange(4))
    with pytest.raises(ValueError):
        training_indices(0, 10, 42)


def test_model_selection_uses_mae_and_stable_ties():
    results = {"ridge": {"mae_minutes": 4.0}, "median": {"mae_minutes": 4.0},
               "hist_gradient_boosting": {"mae_minutes": 5.0}}
    assert select_model(results) == "median"
    with pytest.raises(ValueError):
        select_model({})


def test_all_models_fit_without_accessing_non_feature_columns():
    assert len(FEATURES) == 4
    rng = np.random.default_rng(42)
    x = np.column_stack([
        rng.integers(1, 265, 300),
        rng.integers(0, 24, 300),
        rng.integers(1, 8, 300),
        rng.integers(1, 13, 300),
    ]).astype(float)
    y = 3.0 + 0.2 * x[:, 1] + rng.normal(size=300)
    for model in build_estimators(seed=42).values():
        model.fit(x[:200], y[:200])
        predictions = model.predict(x[200:])
        assert predictions.shape == (100,)
        assert np.isfinite(predictions).all()


def test_hist_boosting_handles_more_than_255_raw_zone_categories():
    # 265 raw zones must be consolidated below the histogram category limit.
    x = np.column_stack([
        np.arange(1, 266),
        np.full(265, 12),
        np.full(265, 2),
        np.full(265, 1),
    ]).astype(float)
    y = np.arange(265, dtype=float) / 10 + 15
    model = build_estimators(42)["hist_gradient_boosting"]
    model.fit(x, y)
    assert np.isfinite(model.predict(x)).all()
    unseen = np.array([[999, 12, 2, 1]], dtype=float)
    assert np.isfinite(model.predict(unseen)).all()
