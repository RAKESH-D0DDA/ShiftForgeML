"""Dependency-independent provenance tests."""

import hashlib
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from shiftforge.contracts import ContractError
from shiftforge.ingestion.provenance import (
    canonical_json_hash,
    sha256_file,
    source_identity,
)


class ProvenanceTests(unittest.TestCase):
    def test_official_source_identity(self):
        year, month, url = source_identity("yellow_tripdata_2024-01.parquet", 2024)
        self.assertEqual((year, month), (2024, 1))
        self.assertTrue(url.endswith("/yellow_tripdata_2024-01.parquet"))
        self.assertTrue(url.startswith("https://"))

    def test_reject_nonbenchmark_year(self):
        with self.assertRaises(ContractError):
            source_identity("yellow_tripdata_2025-01.parquet", 2024)

    def test_reject_invalid_month(self):
        with self.assertRaises(ContractError):
            source_identity("yellow_tripdata_2024-13.parquet", 2024)

    def test_reject_unexpected_source_name(self):
        with self.assertRaises(ContractError):
            source_identity("private_dataset_2024-01.parquet", 2024)

    def test_streaming_hash_matches_reference(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.bin"
            content = b"reproducible binary fixture" * 1024
            path.write_bytes(content)
            self.assertEqual(sha256_file(path), hashlib.sha256(content).hexdigest())

    def test_canonical_hash_ignores_object_key_order(self):
        first = {"settings": {"seed": 42, "sample": 1}, "version": 2}
        second = {"version": 2, "settings": {"sample": 1, "seed": 42}}
        self.assertEqual(canonical_json_hash(first), canonical_json_hash(second))


if __name__ == "__main__":
    unittest.main()
