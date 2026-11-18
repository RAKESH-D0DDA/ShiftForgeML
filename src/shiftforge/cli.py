"""Command-line interface for reproducible benchmark preparation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from shiftforge.contracts import load_spec
from shiftforge.ingestion.parquet import process_month
from shiftforge.ingestion.provenance import source_identity


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="shiftforge")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="validate and prepare a public monthly Parquet file")
    prepare.add_argument("--input", type=Path, required=True, help="downloaded monthly Parquet file")
    prepare.add_argument("--output", type=Path, default=Path("data/processed"))
    prepare.add_argument("--config", type=Path, default=Path("configs/experiment.json"))
    prepare.add_argument("--sample-bps", type=int, default=10_000,
                         help="deterministic sample fraction in basis points (1000 = 10%%)")
    prepare.add_argument("--overwrite", action="store_true", help="replace an existing output")
    args = parser.parse_args(argv)

    spec = load_spec(args.config)
    source_identity(args.input, spec["data_source"]["year"])
    manifest = process_month(
        args.input, args.output, spec,
        sample_bps=args.sample_bps, overwrite=args.overwrite,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
