"""Dataset and evaluation contracts for temporal ML experiments."""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping
import json


class ContractError(ValueError):
    """A dataset or evaluation invariant was violated."""


ALLOWED_FEATURES = frozenset({
    "pickup_zone", "pickup_hour", "pickup_weekday", "pickup_month"
})
REQUIRED_RAW_FIELDS = frozenset({
    "tpep_pickup_datetime", "tpep_dropoff_datetime", "PULocationID"
})
EXPECTED_SPLITS = ("train", "validation", "calibration", "test")


@dataclass(frozen=True)
class TimeWindow:
    name: str
    start: datetime
    end: datetime

    def contains(self, timestamp: datetime) -> bool:
        return self.start <= timestamp < self.end


def _require_naive(timestamp: Any, field: str) -> datetime:
    if not isinstance(timestamp, datetime) or timestamp.tzinfo is not None:
        raise ContractError(f"{field} must be a timezone-naive datetime")
    return timestamp


def validate_spec(spec: Mapping[str, Any]) -> tuple[TimeWindow, ...]:
    """Validate feature availability and non-overlapping chronological splits."""
    features = spec.get("model_features")
    if not isinstance(features, list) or not features:
        raise ContractError("model_features must be a nonempty list")
    if len(features) != len(set(features)) or not set(features) <= ALLOWED_FEATURES:
        raise ContractError("model_features contains duplicates or unavailable features")

    limits = spec.get("target", {})
    low, high = limits.get("minimum_minutes"), limits.get("maximum_minutes")
    if (not isinstance(low, (int, float)) or isinstance(low, bool)
            or not isinstance(high, (int, float)) or isinstance(high, bool)
            or not 0 < low < high):
        raise ContractError("target duration limits are invalid")

    raw_splits = spec.get("splits")
    if not isinstance(raw_splits, list) or len(raw_splits) != len(EXPECTED_SPLITS):
        raise ContractError("expected exactly four temporal splits")
    windows = []
    for raw in raw_splits:
        try:
            window = TimeWindow(
                name=raw["name"],
                start=_require_naive(datetime.fromisoformat(raw["start"]), "start"),
                end=_require_naive(datetime.fromisoformat(raw["end"]), "end"),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ContractError("invalid split definition") from exc
        if window.start >= window.end:
            raise ContractError("split start must precede split end")
        windows.append(window)

    if tuple(w.name for w in windows) != EXPECTED_SPLITS:
        raise ContractError("unexpected split names or order")
    if any(a.end != b.start for a, b in zip(windows, windows[1:])):
        raise ContractError("splits must be contiguous and non-overlapping")
    return tuple(windows)


def load_spec(path: str | Path) -> dict[str, Any]:
    spec = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_spec(spec)
    return spec


def duration_minutes(row: Mapping[str, Any], spec: Mapping[str, Any]) -> float:
    """Validate a raw record and derive the post-trip regression target."""
    missing = REQUIRED_RAW_FIELDS - row.keys()
    if missing:
        raise ContractError(f"missing raw columns: {sorted(missing)}")
    pickup = _require_naive(row["tpep_pickup_datetime"], "pickup")
    dropoff = _require_naive(row["tpep_dropoff_datetime"], "dropoff")
    zone = row["PULocationID"]
    if type(zone) is not int or not 1 <= zone <= 265:
        raise ContractError("invalid pickup zone")
    minutes = (dropoff - pickup).total_seconds() / 60.0
    limits = spec["target"]
    if not limits["minimum_minutes"] <= minutes <= limits["maximum_minutes"]:
        raise ContractError("trip duration is outside the accepted range")
    return minutes


def require_label_available(row: Mapping[str, Any], as_of: datetime) -> None:
    """Reject examples whose targets were unavailable at the evaluation cutoff."""
    cutoff = _require_naive(as_of, "as_of")
    dropoff = _require_naive(row["tpep_dropoff_datetime"], "dropoff")
    if dropoff > cutoff:
        raise ContractError("label was unavailable at the requested cutoff")
