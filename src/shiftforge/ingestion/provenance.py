"""Dependency-independent input validation and provenance utilities."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from shiftforge.contracts import ContractError

SOURCE_PATTERN = re.compile(r"yellow_tripdata_(\d{4})-(\d{2})\.parquet\Z")
SOURCE_ORIGIN = "https://d37ci6vzurychx.cloudfront.net/trip-data"
DATA_CONTRACT_VERSION = "2.0"


def source_identity(path: str | Path, expected_year: int) -> tuple[int, int, str]:
    """Derive and constrain the canonical public source identity."""
    filename = Path(path).name
    match = SOURCE_PATTERN.fullmatch(filename)
    if match is None:
        raise ContractError("source filename must be yellow_tripdata_YYYY-MM.parquet")
    year, month = map(int, match.groups())
    if year != expected_year or not 1 <= month <= 12:
        raise ContractError("source month or year is outside the configured benchmark")
    return year, month, f"{SOURCE_ORIGIN}/{filename}"


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Hash a file without loading it entirely into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_hash(payload: Mapping[str, Any]) -> str:
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path: Path, content: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(content, indent=2, sort_keys=True) + "\n", encoding="utf-8")
