"""End-to-end synthetic monthly ingestion followed by temporal model fitting."""

import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest

pl = pytest.importorskip("polars")

from shiftforge.contracts import ContractError, load_spec
from shiftforge.ingestion.parquet import process_month
from shiftforge.modeling.train import run_training

ROOT = Path(__file__).resolve().parents[2]
SPEC = load_spec(ROOT / "configs/experiment.json")


def prepare_synthetic(tmp_path):
    root = tmp_path / "processed"
    for month in range(1, 9):
        rows = []
        for i in range(45):
            pickup = datetime(2024, month, 1 + (i % 20), i % 24, i % 60)
            minutes = 5 + (i % 20) + (i % 7)
            rows.append((pickup, pickup + timedelta(minutes=minutes), 1 + (i % 30)))
        input_path = tmp_path / f"yellow_tripdata_2024-{month:02d}.parquet"
        pl.DataFrame(rows, schema={
            "tpep_pickup_datetime": pl.Datetime("us"),
            "tpep_dropoff_datetime": pl.Datetime("us"),
            "PULocationID": pl.Int32,
        }, orient="row").write_parquet(input_path)
        process_month(input_path, root, SPEC, sample_bps=10_000)
    return root


def test_training_selects_using_validation_and_records_provenance(tmp_path):
    root = prepare_synthetic(tmp_path)
    report = run_training(ROOT / "configs/experiment.json", root, tmp_path / "artifacts",
                          max_train_rows=70)
    assert report["stage"] == "validation_model_selection"
    assert report["train_rows_available"] == 270
    assert report["train_rows_used"] == 70
    assert report["validation_rows"] == 90
    assert report["train_months"] == [f"2024-{m:02d}" for m in range(1, 7)]
    assert report["validation_months"] == ["2024-07", "2024-08"]
    assert set(report["validation_metrics"]) == {"median", "ridge", "hist_gradient_boosting"}
    assert report["selected_model"] in report["validation_metrics"]
    assert len(report["curated_inputs"]) == 8
    artifact = tmp_path / "artifacts" / "selected_model.joblib"
    assert hashlib.sha256(artifact.read_bytes()).hexdigest() == report["model_sha256"]
    stored = json.loads((tmp_path / "artifacts/validation_metrics.json").read_text())
    assert stored["selected_model"] == report["selected_model"]
    with pytest.raises(FileExistsError):
        run_training(ROOT / "configs/experiment.json", root, tmp_path / "artifacts",
                     max_train_rows=70)


def test_inconsistent_sampling_manifest_is_rejected(tmp_path):
    root = prepare_synthetic(tmp_path)
    manifest_path = root / "manifests/2024-08.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["sample_basis_points"] = 3000
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ContractError, match="sampling rates differ"):
        run_training(ROOT / "configs/experiment.json", root, tmp_path / "output")


def test_corrupted_curated_partition_rejected(tmp_path):
    root = prepare_synthetic(tmp_path)
    target = root / "curated/2024-03.parquet"
    target.write_bytes(target.read_bytes() + b"altered")
    with pytest.raises(ContractError, match="checksum mismatch"):
        run_training(ROOT / "configs/experiment.json", root, tmp_path / "output")
