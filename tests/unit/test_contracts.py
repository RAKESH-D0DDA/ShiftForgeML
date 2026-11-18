"""Unit tests for source-data and temporal evaluation contracts."""

import copy
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from shiftforge.contracts import (
    ContractError, duration_minutes, load_spec, require_label_available,
    validate_spec,
)

ROOT = Path(__file__).resolve().parents[2]


class ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = load_spec(ROOT / "configs/experiment.json")

    def sample_row(self):
        return {
            "tpep_pickup_datetime": datetime(2024, 6, 30, 23, 50),
            "tpep_dropoff_datetime": datetime(2024, 7, 1, 0, 5),
            "PULocationID": 132,
        }

    def test_temporal_windows_are_disjoint(self):
        windows = validate_spec(self.spec)
        self.assertTrue(windows[0].contains(datetime(2024, 6, 30, 23, 59)))
        self.assertFalse(windows[0].contains(datetime(2024, 7, 1)))
        self.assertTrue(windows[1].contains(datetime(2024, 7, 1)))

    def test_overlapping_windows_rejected(self):
        modified = copy.deepcopy(self.spec)
        modified["splits"][1]["start"] = "2024-06-30T00:00:00"
        with self.assertRaises(ContractError):
            validate_spec(modified)

    def test_post_outcome_feature_rejected(self):
        modified = copy.deepcopy(self.spec)
        modified["model_features"].append("trip_distance")
        with self.assertRaises(ContractError):
            validate_spec(modified)

    def test_duplicate_feature_rejected(self):
        modified = copy.deepcopy(self.spec)
        modified["model_features"].append("pickup_zone")
        with self.assertRaises(ContractError):
            validate_spec(modified)

    def test_duration_computed_in_minutes(self):
        self.assertEqual(duration_minutes(self.sample_row(), self.spec), 15.0)

    def test_negative_duration_rejected(self):
        row = self.sample_row()
        row["tpep_dropoff_datetime"] = row["tpep_pickup_datetime"] - timedelta(minutes=1)
        with self.assertRaises(ContractError):
            duration_minutes(row, self.spec)

    def test_missing_raw_field_rejected(self):
        row = self.sample_row()
        del row["PULocationID"]
        with self.assertRaises(ContractError):
            duration_minutes(row, self.spec)

    def test_invalid_zone_rejected(self):
        row = self.sample_row()
        row["PULocationID"] = 0
        with self.assertRaises(ContractError):
            duration_minutes(row, self.spec)

    def test_future_label_rejected(self):
        with self.assertRaises(ContractError):
            require_label_available(self.sample_row(), datetime(2024, 7, 1))

    def test_available_label_accepted(self):
        require_label_available(self.sample_row(), datetime(2024, 7, 1, 0, 5))

    def test_timezone_aware_datetime_rejected(self):
        row = self.sample_row()
        row["tpep_pickup_datetime"] = datetime(2024, 6, 30, tzinfo=timezone.utc)
        with self.assertRaises(ContractError):
            duration_minutes(row, self.spec)


if __name__ == "__main__":
    unittest.main()
