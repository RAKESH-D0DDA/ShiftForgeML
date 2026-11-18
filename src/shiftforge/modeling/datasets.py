"""Manifest-verified chronological data access for supervised experiments."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from shiftforge.contracts import ContractError, EXPECTED_SPLITS, validate_spec
from shiftforge.ingestion.provenance import DATA_CONTRACT_VERSION, canonical_json_hash, sha256_file
from shiftforge.modeling.estimators import FEATURES


@dataclass(frozen=True)
class SplitData:
    features: np.ndarray
    targets: np.ndarray
    months: tuple[str, ...]
    input_manifests: tuple[dict[str, Any], ...]
    sample_basis_points: int


def expected_months(spec: Mapping[str, Any], split: str) -> tuple[str, ...]:
    """Return the complete list of calendar months in an aligned split."""
    windows = validate_spec(spec)
    if split not in EXPECTED_SPLITS:
        raise ContractError(f"unrecognized split: {split}")
    window = next(w for w in windows if w.name == split)
    for boundary in (window.start, window.end):
        if boundary != datetime(boundary.year, boundary.month, 1):
            raise ContractError("model training requires month-aligned split boundaries")
    year, month = window.start.year, window.start.month
    months = []
    while (year, month) < (window.end.year, window.end.month):
        months.append(f"{year:04d}-{month:02d}")
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    if not months:
        raise ContractError("a split must contain at least one month")
    return tuple(months)


def verified_paths(
    processed_root: Path, spec: Mapping[str, Any], split: str,
) -> tuple[list[Path], list[dict[str, Any]], int]:
    """Verify all required partition manifests and their file checksums."""
    expected_digest = canonical_json_hash(spec)
    paths: list[Path] = []
    manifests: list[dict[str, Any]] = []
    sampling: set[int] = set()
    for month in expected_months(spec, split):
        manifest_path = processed_root / "manifests" / f"{month}.json"
        curated_path = processed_root / "curated" / f"{month}.parquet"
        if not manifest_path.is_file() or not curated_path.is_file():
            raise FileNotFoundError(f"missing curated partition or manifest for {month}")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (ValueError, OSError) as exc:
            raise ContractError(f"invalid manifest for {month}") from exc
        if (
            manifest.get("source_month") != month
            or manifest.get("curated_filename") != curated_path.name
            or manifest.get("contract_version") != DATA_CONTRACT_VERSION
            or manifest.get("experiment_sha256") != expected_digest
            or manifest.get("experiment_id") != spec["experiment_id"]
        ):
            raise ContractError(f"manifest configuration mismatch for {month}")
        if manifest.get("curated_sha256") != sha256_file(curated_path):
            raise ContractError(f"curated file checksum mismatch for {month}")
        count = manifest.get("sampled_rows")
        if type(count) is not int or count < 0:
            raise ContractError(f"invalid sampled row count for {month}")
        split_count = manifest.get("sampled_by_split", {}).get(split, 0)
        if split_count != count:
            raise ContractError(f"split count mismatch for {month}")
        sample_bps = manifest.get("sample_basis_points")
        if type(sample_bps) is not int or not 1 <= sample_bps <= 10_000:
            raise ContractError(f"invalid sampling configuration for {month}")
        sampling.add(sample_bps)
        paths.append(curated_path)
        manifests.append({
            "month": month,
            "curated_sha256": manifest["curated_sha256"],
            "input_sha256": manifest.get("input_sha256"),
            "sampled_rows": count,
            "sample_basis_points": sample_bps,
        })
    if len(sampling) != 1:
        raise ContractError("all model partitions must use the same sampling fraction")
    return paths, manifests, sampling.pop()


def load_split(processed_root: str | Path, spec: Mapping[str, Any], split: str) -> SplitData:
    """Load an entire split after verifying provenance and temporal invariants.

    This boundary materializes NumPy arrays in memory. It is intended for the
    baseline estimator suite, not for unlimited out-of-core training.
    """
    import polars as pl

    if tuple(spec["model_features"]) != FEATURES:
        raise ContractError("model feature order must match the versioned feature contract")
    window = next(w for w in validate_spec(spec) if w.name == split)
    paths, manifests, sample_bps = verified_paths(Path(processed_root), spec, split)
    required = (*FEATURES, "duration_minutes", "pickup_time", "label_available_at", "split")
    scan = pl.scan_parquet([str(p) for p in paths])
    if not set(required) <= set(scan.collect_schema().names()):
        raise ContractError("curated dataset is missing required model columns")
    frame = scan.select(list(required)).collect(engine="streaming")
    if frame.height != sum(m["sampled_rows"] for m in manifests):
        raise ContractError("Parquet row count differs from the manifest")
    if frame.is_empty():
        raise ContractError(f"split contains no observations: {split}")

    nulls = pl.any_horizontal(*(pl.col(col).is_null() for col in required))
    bad = (
        nulls
        | (pl.col("split") != split)
        | ~pl.col("pickup_time").is_between(window.start, window.end, closed="left")
        | (pl.col("label_available_at") < pl.col("pickup_time"))
        | (pl.col("label_available_at") > window.end)
        | ~pl.col("pickup_zone").is_between(1, 265)
        | ~pl.col("pickup_hour").is_between(0, 23)
        | ~pl.col("pickup_weekday").is_between(1, 7)
        | ~pl.col("pickup_month").is_between(1, 12)
        | (pl.col("pickup_hour") != pl.col("pickup_time").dt.hour())
        | (pl.col("pickup_weekday") != pl.col("pickup_time").dt.weekday())
        | (pl.col("pickup_month") != pl.col("pickup_time").dt.month())
        | ~pl.col("duration_minutes").is_finite()
        | ~pl.col("duration_minutes").is_between(
            spec["target"]["minimum_minutes"], spec["target"]["maximum_minutes"]
        )
    )
    if frame.filter(bad).height:
        raise ContractError(f"curated data violates temporal or feature contracts: {split}")
    x = frame.select(list(FEATURES)).to_numpy().astype(np.float64, copy=False)
    y = frame.get_column("duration_minutes").to_numpy().astype(np.float64, copy=False)
    return SplitData(x, y, expected_months(spec, split), tuple(manifests), sample_bps)
