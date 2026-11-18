"""Chronological partition contracts independent of Parquet dependencies."""

from pathlib import Path
import copy

import pytest

from shiftforge.contracts import ContractError, load_spec
from shiftforge.modeling.datasets import expected_months, verified_paths

ROOT = Path(__file__).resolve().parents[2]
SPEC = load_spec(ROOT / "configs/experiment.json")


def test_expected_months_exact():
    assert expected_months(SPEC, "train") == tuple(f"2024-{m:02d}" for m in range(1, 7))
    assert expected_months(SPEC, "validation") == ("2024-07", "2024-08")
    assert expected_months(SPEC, "calibration") == ("2024-09",)
    assert expected_months(SPEC, "test") == ("2024-10", "2024-11", "2024-12")


def test_non_aligned_boundaries_rejected():
    modified = copy.deepcopy(SPEC)
    modified["splits"][0]["end"] = "2024-07-02T00:00:00"
    modified["splits"][1]["start"] = "2024-07-02T00:00:00"
    with pytest.raises(ContractError, match="month-aligned"):
        expected_months(modified, "train")


def test_missing_month_fails_before_training(tmp_path):
    with pytest.raises(FileNotFoundError, match="2024-01"):
        verified_paths(tmp_path, SPEC, "train")
