"""End-to-end Parquet tests with generated synthetic records only."""

import json
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from shiftforge.contracts import ContractError, load_spec
from shiftforge.ingestion.parquet import OUTPUT_COLUMNS, process_month

ROOT = Path(__file__).resolve().parents[2]
SPEC = load_spec(ROOT / "configs/experiment.json")


def make_source(tmp_path, rows, *, name="yellow_tripdata_2024-01.parquet"):
    path = tmp_path / name
    # Explicit datatypes keep null-heavy test fixtures schema-compatible.
    pl.DataFrame(
        rows,
        schema={
            "tpep_pickup_datetime": pl.Datetime("us"),
            "tpep_dropoff_datetime": pl.Datetime("us"),
            "PULocationID": pl.Int32,
        },
        orient="row",
    ).write_parquet(path)
    return path


def make_rows():
    return [
        (datetime(2024, 1, 5, 10), datetime(2024, 1, 5, 10, 15), 132),
        (datetime(2024, 1, 8, 16), datetime(2024, 1, 8, 16, 31), 161),
        (datetime(2024, 1, 12, 12), datetime(2024, 1, 12, 12, 5), 1),
        (datetime(2024, 1, 15, 9), datetime(2024, 1, 15, 9, 15), 0),
        (datetime(2024, 1, 17, 8), datetime(2024, 1, 17, 8, 0, 30), 42),
        (datetime(2024, 1, 18, 8), datetime(2024, 1, 18, 8, 20), None),
        (datetime(2024, 2, 1, 8), datetime(2024, 2, 1, 8, 12), 132),
        (datetime(2024, 1, 31, 23, 50), datetime(2024, 2, 1, 0, 20), 132),
        (datetime(2024, 1, 19, 8), datetime(2024, 1, 19, 7, 55), 55),
    ]


def test_ingestion_schema_and_rejection_accounting(tmp_path):
    source = make_source(tmp_path, make_rows())
    manifest = process_month(source, tmp_path / "output", SPEC)
    curated = pl.read_parquet(tmp_path / "output/curated/2024-01.parquet")
    assert curated.columns == list(OUTPUT_COLUMNS)
    assert manifest["rows_scanned"] == 9
    assert manifest["accepted_before_sampling"] == 4
    assert manifest["sampled_rows"] == 4
    assert manifest["rejected_by_reason"] == {
        "invalid_duration": 2,
        "invalid_zone": 1,
        "missing_required": 1,
        "outside_source_month": 1,
    }
    assert sorted(curated["duration_minutes"].to_list()) == [5.0, 15.0, 30.0, 31.0]
    assert set(curated["split"]) == {"train"}
    assert (curated["pickup_weekday"] >= 1).all()
    assert (curated["pickup_weekday"] <= 7).all()
    assert (curated["label_available_at"] >= curated["pickup_time"]).all()


def test_manifest_is_cached_for_identical_inputs(tmp_path):
    source = make_source(tmp_path, make_rows())
    output = tmp_path / "output"
    first = process_month(source, output, SPEC)
    second = process_month(source, output, SPEC)
    assert first["execution_status"] == "processed"
    assert second["execution_status"] == "cache_hit"
    assert first["curated_sha256"] == second["curated_sha256"]
    assert (output / "manifests/2024-01.json").is_file()


def test_overwrite_required_when_sampling_changes(tmp_path):
    source = make_source(tmp_path, make_rows())
    output = tmp_path / "output"
    process_month(source, output, SPEC)
    with pytest.raises(FileExistsError):
        process_month(source, output, SPEC, sample_bps=5000)
    updated = process_month(source, output, SPEC, sample_bps=5000, overwrite=True)
    assert updated["sample_basis_points"] == 5000


def test_sampling_is_repeatable_and_seeded(tmp_path):
    # Generated records span many different pickup timestamps.
    rows = [
        (datetime(2024, 1, 4, 8, i % 60, i // 60),
         datetime(2024, 1, 4, 8, i % 60, i // 60) + timedelta(minutes=15),
         (i % 100) + 1)
        for i in range(300)
    ]
    source = make_source(tmp_path, rows)
    first = process_month(source, tmp_path / "first", SPEC, sample_bps=3000)
    second = process_month(source, tmp_path / "second", SPEC, sample_bps=3000)
    assert 0 < first["sampled_rows"] < first["accepted_before_sampling"]
    assert first["curated_sha256"] == second["curated_sha256"]


def test_missing_required_column_is_rejected(tmp_path):
    source = tmp_path / "yellow_tripdata_2024-01.parquet"
    pl.DataFrame({"PULocationID": [1]}).write_parquet(source)
    with pytest.raises(ContractError, match="missing raw columns"):
        process_month(source, tmp_path / "output", SPEC)


def test_late_labels_excluded_at_training_cutoff(tmp_path):
    rows = [
        (datetime(2024, 6, 30, 23, 59), datetime(2024, 7, 1, 0, 5), 132),
        (datetime(2024, 6, 30, 23, 40), datetime(2024, 6, 30, 23, 55), 132),
    ]
    source = make_source(tmp_path, rows, name="yellow_tripdata_2024-06.parquet")
    manifest = process_month(source, tmp_path / "output", SPEC)
    assert manifest["rejected_by_reason"] == {"label_unavailable_at_split_end": 1}
    assert manifest["sampled_rows"] == 1


def test_manifest_row_counts_are_consistent(tmp_path):
    source = make_source(tmp_path, make_rows())
    manifest = process_month(source, tmp_path / "output", SPEC)
    assert (
        manifest["rows_scanned"]
        == manifest["accepted_before_sampling"] + sum(manifest["rejected_by_reason"].values())
    )
    stored = json.loads((tmp_path / "output/manifests/2024-01.json").read_text())
    assert stored["curated_sha256"] == manifest["curated_sha256"]
