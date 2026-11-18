"""Fit three regressors on training data and select using validation MAE."""

from __future__ import annotations

import json
import os
from pathlib import Path
import platform
import tempfile
from time import perf_counter
from typing import Any

import joblib
import numpy as np
import sklearn

from shiftforge.contracts import ContractError, load_spec
from shiftforge.ingestion.provenance import canonical_json_hash, sha256_file, utc_timestamp, write_json
from shiftforge.modeling.datasets import load_split
from shiftforge.modeling.estimators import FEATURES, build_estimators
from shiftforge.modeling.metrics import regression_metrics, select_model, training_indices


def run_training(
    config: str | Path, processed_root: str | Path, output_dir: str | Path,
    *, max_train_rows: int = 200_000, overwrite: bool = False,
) -> dict[str, Any]:
    """Select on validation only. Never read calibration or test partitions."""
    spec = load_spec(config)
    if type(max_train_rows) is not int or max_train_rows < 10:
        raise ContractError("max_train_rows must be an integer of at least 10")
    output_dir = Path(output_dir)
    model_path = output_dir / "selected_model.joblib"
    report_path = output_dir / "validation_metrics.json"
    if not overwrite and (model_path.exists() or report_path.exists()):
        raise FileExistsError("training artifacts already exist; specify --overwrite")

    train = load_split(processed_root, spec, "train")
    validation = load_split(processed_root, spec, "validation")
    if train.sample_basis_points != validation.sample_basis_points:
        raise ContractError("training and validation sampling rates differ")
    if len(train.targets) < 10 or len(validation.targets) < 10:
        raise ContractError("each partition must contain at least ten observations")
    seed = spec["random_seed"]
    indices = training_indices(len(train.targets), max_train_rows, seed)
    x_train, y_train = train.features[indices], train.targets[indices]

    fitted = {}
    results: dict[str, dict[str, float | int]] = {}
    for name, model in build_estimators(seed).items():
        start = perf_counter()
        model.fit(x_train, y_train)
        duration = perf_counter() - start
        prediction = model.predict(validation.features)
        metrics = regression_metrics(validation.targets, prediction)
        metrics["fit_seconds"] = round(duration, 6)
        results[name] = metrics
        fitted[name] = model
    selected = select_model(results)

    output_dir.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=".model-", suffix=".joblib", dir=output_dir)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        joblib.dump(fitted[selected], temporary, compress=3)
        report: dict[str, Any] = {
            "stage": "validation_model_selection",
            "experiment_id": spec["experiment_id"],
            "experiment_sha256": canonical_json_hash(spec),
            "created_at_utc": utc_timestamp(),
            "features": list(FEATURES),
            "random_seed": seed,
            "sampling_basis_points": train.sample_basis_points,
            "train_months": list(train.months),
            "validation_months": list(validation.months),
            "train_rows_available": len(train.targets),
            "train_rows_used": len(y_train),
            "validation_rows": len(validation.targets),
            "max_train_rows": max_train_rows,
            "selection_metric": "validation_mae_minutes",
            "selected_model": selected,
            "validation_metrics": results,
            "curated_inputs": [*train.input_manifests, *validation.input_manifests],
            "model_artifact": model_path.name,
            "model_sha256": sha256_file(temporary),
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "sklearn_version": sklearn.__version__,
        }
        os.replace(temporary, model_path)
        temporary_json = output_dir / ".validation_metrics.json.tmp"
        try:
            write_json(temporary_json, report)
            os.replace(temporary_json, report_path)
        finally:
            temporary_json.unlink(missing_ok=True)
        return report
    finally:
        temporary.unlink(missing_ok=True)
