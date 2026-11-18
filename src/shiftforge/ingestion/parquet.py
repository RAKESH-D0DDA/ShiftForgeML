"""Schema-checked, lazy Parquet ingestion with deterministic sampling.

Only the three permitted raw columns are read. The output contains an
explicit target and label-availability timestamp, neither of which may be
used as a model input.
"""

from __future__ import annotations

import json
import os
import platform
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import polars as pl

from shiftforge.contracts import ContractError, REQUIRED_RAW_FIELDS, validate_spec
from shiftforge.ingestion.provenance import (
    DATA_CONTRACT_VERSION,
    canonical_json_hash,
    sha256_file,
    source_identity,
    utc_timestamp,
    write_json,
)

OUTPUT_COLUMNS = (
    "pickup_time", "label_available_at", "pickup_zone", "pickup_hour",
    "pickup_weekday", "pickup_month", "duration_minutes", "split",
)


def _input_schema(frame: pl.LazyFrame) -> dict[str, Any]:
    schema = dict(frame.collect_schema())
    missing = REQUIRED_RAW_FIELDS - schema.keys()
    if missing:
        raise ContractError(f"missing raw columns: {sorted(missing)}")
    for field in ("tpep_pickup_datetime", "tpep_dropoff_datetime"):
        dtype = schema[field]
        if not isinstance(dtype, pl.Datetime) or dtype.time_zone is not None:
            raise ContractError(f"{field} must be a timezone-naive timestamp")
    zone_dtype = schema["PULocationID"]
    if zone_dtype not in (
        pl.Int8, pl.Int16, pl.Int32, pl.Int64,
        pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
    ):
        raise ContractError("PULocationID must have an integer type")
    return schema


def _month_bounds(year: int, month: int) -> tuple[datetime, datetime]:
    start = datetime(year, month, 1)
    end = datetime(year + 1, 1, 1) if month == 12 else datetime(year, month + 1, 1)
    return start, end


def _split_expression(windows: tuple) -> pl.Expr:
    expression = pl.lit(None, dtype=pl.String)
    pickup = pl.col("tpep_pickup_datetime")
    for window in reversed(windows):
        expression = (
            pl.when((pickup >= window.start) & (pickup < window.end))
            .then(pl.lit(window.name))
            .otherwise(expression)
        )
    return expression.alias("_split")


def _cutoff_expression(windows: tuple) -> pl.Expr:
    expression = pl.lit(None, dtype=pl.Datetime("us"))
    for window in reversed(windows):
        expression = (
            pl.when(pl.col("_split") == window.name)
            .then(pl.lit(window.end))
            .otherwise(expression)
        )
    return expression.alias("_label_cutoff")


def _classified(
    source: Path, spec: Mapping[str, Any], year: int, month: int,
) -> pl.LazyFrame:
    windows = validate_spec(spec)
    input_frame = pl.scan_parquet(str(source))
    _input_schema(input_frame)
    pickup = pl.col("tpep_pickup_datetime")
    dropoff = pl.col("tpep_dropoff_datetime")
    zone = pl.col("PULocationID")
    month_start, month_end = _month_bounds(year, month)
    limits = spec["target"]

    frame = (
        input_frame.select(sorted(REQUIRED_RAW_FIELDS))
        .with_columns(
            ((dropoff - pickup).dt.total_microseconds() / 60_000_000.0)
            .alias("_duration_minutes"),
            _split_expression(windows),
        )
        .with_columns(_cutoff_expression(windows))
    )
    missing = pl.any_horizontal(pickup.is_null(), dropoff.is_null(), zone.is_null())
    reason = (
        pl.when(missing).then(pl.lit("missing_required"))
        .when((pickup < month_start) | (pickup >= month_end))
        .then(pl.lit("outside_source_month"))
        .when(pl.col("_split").is_null()).then(pl.lit("outside_experiment"))
        .when(~zone.is_between(1, 265)).then(pl.lit("invalid_zone"))
        .when(~pl.col("_duration_minutes").is_between(
            limits["minimum_minutes"], limits["maximum_minutes"]
        )).then(pl.lit("invalid_duration"))
        .when(dropoff > pl.col("_label_cutoff"))
        .then(pl.lit("label_unavailable_at_split_end"))
        .otherwise(pl.lit("accepted"))
        .alias("_reason")
    )
    return frame.with_columns(reason)


def _sample_filter(seed: int, basis_points: int) -> pl.Expr:
    # Reproducible within the recorded Polars version. No row-order dependence.
    row_hash = pl.struct(
        ["tpep_pickup_datetime", "tpep_dropoff_datetime", "PULocationID"]
    ).hash(seed=seed)
    return (row_hash % 10_000) < basis_points


def _curated(frame: pl.LazyFrame, seed: int, basis_points: int) -> pl.LazyFrame:
    pickup = pl.col("tpep_pickup_datetime")
    return (
        frame.filter(pl.col("_reason") == "accepted")
        .filter(_sample_filter(seed, basis_points))
        .select(
            pickup.alias("pickup_time"),
            pl.col("tpep_dropoff_datetime").alias("label_available_at"),
            pl.col("PULocationID").cast(pl.Int16).alias("pickup_zone"),
            pickup.dt.hour().cast(pl.Int8).alias("pickup_hour"),
            pickup.dt.weekday().cast(pl.Int8).alias("pickup_weekday"),
            pickup.dt.month().cast(pl.Int8).alias("pickup_month"),
            pl.col("_duration_minutes").cast(pl.Float64).alias("duration_minutes"),
            pl.col("_split").alias("split"),
        )
    )


def _counts(frame: pl.LazyFrame) -> tuple[dict[str, int], dict[str, int], int]:
    grouped = (
        frame.group_by(["_reason", "_split"])
        .agg(pl.len().alias("n"))
        .collect(engine="streaming")
    )
    by_reason: dict[str, int] = {}
    accepted_by_split: dict[str, int] = {}
    total = 0
    for row in grouped.iter_rows(named=True):
        reason, split, n = row["_reason"], row["_split"], row["n"]
        n = int(n)
        by_reason[reason] = by_reason.get(reason, 0) + n
        if reason == "accepted" and split is not None:
            accepted_by_split[split] = accepted_by_split.get(split, 0) + n
        total += n
    return dict(sorted(by_reason.items())), dict(sorted(accepted_by_split.items())), total


def _output_counts(path: Path) -> dict[str, int]:
    grouped = (
        pl.scan_parquet(str(path)).group_by("split")
        .agg(pl.len().alias("n"))
        .collect(engine="streaming")
    )
    return dict(sorted((str(split), int(n)) for split, n in grouped.iter_rows()))


def _cached_manifest(
    manifest_path: Path, curated_path: Path, input_digest: str,
    spec_digest: str, sample_bps: int, polars_version: str,
) -> dict[str, Any] | None:
    if not (manifest_path.exists() and curated_path.exists()):
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if (
        manifest.get("input_sha256") != input_digest
        or manifest.get("experiment_sha256") != spec_digest
        or manifest.get("sample_basis_points") != sample_bps
        or manifest.get("polars_version") != polars_version
        or manifest.get("contract_version") != DATA_CONTRACT_VERSION
    ):
        return None
    if manifest.get("curated_sha256") != sha256_file(curated_path):
        return None
    manifest["execution_status"] = "cache_hit"
    return manifest


def process_month(
    source: str | Path, output_root: str | Path, spec: Mapping[str, Any],
    *, sample_bps: int = 10_000, overwrite: bool = False,
) -> dict[str, Any]:
    """Produce a monthly curated Parquet file and a versioned provenance manifest."""
    if type(sample_bps) is not int or not 1 <= sample_bps <= 10_000:
        raise ContractError("sample_bps must be an integer between 1 and 10000")
    windows = validate_spec(spec)
    source = Path(source).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"input Parquet file not found: {source}")
    year, month, source_url = source_identity(source, spec["data_source"]["year"])
    output_root = Path(output_root).expanduser().resolve()
    curated_path = output_root / "curated" / f"{year}-{month:02d}.parquet"
    manifest_path = output_root / "manifests" / f"{year}-{month:02d}.json"
    curated_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    input_digest = sha256_file(source)
    spec_digest = canonical_json_hash(spec)
    if not overwrite:
        cached = _cached_manifest(
            manifest_path, curated_path, input_digest,
            spec_digest, sample_bps, pl.__version__,
        )
        if cached is not None:
            return cached
        if curated_path.exists() or manifest_path.exists():
            raise FileExistsError("existing output differs from inputs or is incomplete; use --overwrite")

    frame = _classified(source, spec, year, month)
    by_reason, accepted_by_split, scanned_count = _counts(frame)

    # A temporary output avoids replacing a valid dataset after a failed scan.
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{year}-{month:02d}-", suffix=".parquet", dir=curated_path.parent
    )
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)
    try:
        _curated(frame, spec["random_seed"], sample_bps).sink_parquet(
            temporary_path, compression="zstd", maintain_order=True
        )
        output_by_split = _output_counts(temporary_path)
        output_count = sum(output_by_split.values())
        accepted_count = by_reason.get("accepted", 0)
        if output_count > accepted_count:
            raise RuntimeError("curated count exceeds accepted count")
        manifest: dict[str, Any] = {
            "contract_version": DATA_CONTRACT_VERSION,
            "experiment_id": spec["experiment_id"],
            "experiment_sha256": spec_digest,
            "created_at_utc": utc_timestamp(),
            "source_url": source_url,
            "source_filename": source.name,
            "source_month": f"{year}-{month:02d}",
            "input_bytes": source.stat().st_size,
            "input_sha256": input_digest,
            "curated_filename": curated_path.name,
            "curated_sha256": sha256_file(temporary_path),
            "polars_version": pl.__version__,
            "python_version": platform.python_version(),
            "random_seed": spec["random_seed"],
            "sample_basis_points": sample_bps,
            "sampled_rows": output_count,
            "accepted_before_sampling": accepted_count,
            "excluded_by_sampling": accepted_count - output_count,
            "rows_scanned": scanned_count,
            "rejected_by_reason": {
                key: n for key, n in by_reason.items() if key != "accepted"
            },
            "accepted_by_split": accepted_by_split,
            "sampled_by_split": output_by_split,
            "output_columns": list(OUTPUT_COLUMNS),
            "split_policy": "pickup_time half-open windows; label available by split end",
            "timestamp_policy": "source timezone-naive; wall-clock subtraction",
        }
        # Persist the data before recording its checksum in the final manifest.
        os.replace(temporary_path, curated_path)
        temp_manifest = manifest_path.with_suffix(".json.tmp")
        try:
            write_json(temp_manifest, manifest)
            os.replace(temp_manifest, manifest_path)
        finally:
            temp_manifest.unlink(missing_ok=True)
        manifest["execution_status"] = "processed"
        return manifest
    finally:
        temporary_path.unlink(missing_ok=True)
