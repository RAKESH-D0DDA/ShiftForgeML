"""Training orchestration without external data dependencies."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from shiftforge.modeling.datasets import SplitData
from shiftforge.modeling import train as training_module

ROOT = Path(__file__).resolve().parents[2]


def test_runner_selects_validation_only_and_saves_verified_artifact(tmp_path, monkeypatch):
    rng = np.random.default_rng(18)
    train_x = np.column_stack([
        rng.integers(1, 240, 110), rng.integers(0, 24, 110),
        rng.integers(1, 8, 110), rng.integers(1, 7, 110),
    ]).astype(float)
    val_x = np.column_stack([
        rng.integers(1, 240, 35), rng.integers(0, 24, 35),
        rng.integers(1, 8, 35), rng.integers(7, 9, 35),
    ]).astype(float)
    train_data = SplitData(train_x, 5 + train_x[:, 1] * 0.7,
                           tuple(f"2024-{m:02d}" for m in range(1, 7)), (), 1000)
    val_data = SplitData(val_x, 5 + val_x[:, 1] * 0.7,
                         ("2024-07", "2024-08"), (), 1000)
    seen = []

    def fake_loader(root, spec, split):
        seen.append(split)
        return {"train": train_data, "validation": val_data}[split]

    monkeypatch.setattr(training_module, "load_split", fake_loader)
    output = tmp_path / "artifacts"
    report = training_module.run_training(ROOT / "configs/experiment.json", tmp_path,
                                           output, max_train_rows=65)
    assert seen == ["train", "validation"]
    assert report["train_rows_used"] == 65
    assert report["validation_rows"] == 35
    assert report["selected_model"] in report["validation_metrics"]
    artifact = output / "selected_model.joblib"
    assert hashlib.sha256(artifact.read_bytes()).hexdigest() == report["model_sha256"]
    assert json.loads((output / "validation_metrics.json").read_text())["stage"] == "validation_model_selection"
    with pytest.raises(FileExistsError):
        training_module.run_training(ROOT / "configs/experiment.json", tmp_path,
                                     output, max_train_rows=65)
