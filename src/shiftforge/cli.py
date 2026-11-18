"""CLI for reproducible ingestion and temporal model selection."""

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

    prepare = commands.add_parser("prepare", help="validate and prepare a monthly Parquet file")
    prepare.add_argument("--input", type=Path, required=True)
    prepare.add_argument("--output", type=Path, default=Path("data/processed"))
    prepare.add_argument("--config", type=Path, default=Path("configs/experiment.json"))
    prepare.add_argument("--sample-bps", type=int, default=10_000,
                         help="deterministic fraction in basis points (1000 = 10%%)")
    prepare.add_argument("--overwrite", action="store_true")

    train = commands.add_parser("train", help="fit and select supervised regression models")
    train.add_argument("--config", type=Path, default=Path("configs/experiment.json"))
    train.add_argument("--processed-root", type=Path, default=Path("data/processed"))
    train.add_argument("--output", type=Path, default=Path("artifacts/training"))
    train.add_argument("--max-train-rows", type=int, default=200_000)
    train.add_argument("--overwrite", action="store_true")

    args = parser.parse_args(argv)
    if args.command == "prepare":
        spec = load_spec(args.config)
        source_identity(args.input, spec["data_source"]["year"])
        result = process_month(
            args.input, args.output, spec,
            sample_bps=args.sample_bps, overwrite=args.overwrite,
        )
    else:
        # Import large ML dependencies only when the training command is invoked.
        from shiftforge.modeling.train import run_training
        result = run_training(
            args.config, args.processed_root, args.output,
            max_train_rows=args.max_train_rows, overwrite=args.overwrite,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
