# ShiftForgeML

ShiftForgeML is an independent experimental framework for temporal machine learning, distribution shift detection, uncertainty quantification and adaptive retraining. The first benchmark uses publicly available taxi trip records to evaluate general ML reliability under changing data distributions.

## Implementation status

**Milestone 1: complete.** Temporal evaluation contracts, feature restrictions and label-availability tests.

**Milestone 2: implemented for local verification.** Lazy Parquet ingestion, deterministic sampling, leakage-aware feature extraction, row-level rejection accounting, cryptographic provenance manifests and synthetic integration tests.

**Not yet implemented:** fitted regression models, optimization, drift detectors, conformal prediction, adaptive retraining and experiment tracking.

## Requirements

Python 3.11 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
python -m pytest -q
```

## Quick start: public benchmark data

The official source provides the 2024 yellow taxi trip records in monthly Parquet format:
https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page

Download the January partition from its public source:

```bash
mkdir -p data/raw
curl --fail --location --retry 3 \
  'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet' \
  --output data/raw/yellow_tripdata_2024-01.parquet
```

Prepare a deterministic 10% sample:

```bash
shiftforge prepare \
  --config configs/experiment.json \
  --input data/raw/yellow_tripdata_2024-01.parquet \
  --output data/processed \
  --sample-bps 1000
```

Output files:

- `data/processed/curated/2024-01.parquet`: cleaned features, target and split assignment
- `data/processed/manifests/2024-01.json`: input and output checksums, processing parameters and rejection counts

`--sample-bps 10000` retains all accepted rows. A repeated identical invocation returns a cache hit. Use `--overwrite` after changing the sample rate or the source data.

## Technical documentation

- [`docs/methodology.md`](docs/methodology.md): temporal evaluation and target-leakage constraints
- [`docs/ingestion.md`](docs/ingestion.md): schema, sampling, provenance and validity policy
- [`configs/experiment.json`](configs/experiment.json): versioned experiment configuration

All input data and generated artifacts stay outside version control. The source is public and the project does not require external services or credentials.
