# ShiftForgeML

ShiftForgeML is an independent framework for temporal machine learning, distribution shift detection, uncertainty quantification and adaptive retraining. Its initial benchmark uses publicly available monthly taxi trip records. The purpose of the project is to evaluate predictive systems under changes in data distribution, not to develop a domain-specific application.

## Current implementation

- **Milestone 1 complete:** temporal data contracts, feature-availability rules and label-cutoff tests.
- **Milestone 2 complete:** Polars-based Parquet ingestion, deterministic sampling, provenance manifests and record-level quality accounting.
- **Milestone 3 implemented, pending independent end-to-end verification:** supervised regression baselines, training-only preprocessing, manifest-verified temporal model selection and validation artifacts.
- **Planned:** drift detection, conformal uncertainty estimation, adaptive retraining and expanded experiment tracking.

## Environment

Python 3.11 or newer. Create a local virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest -q
```

## Data preparation

The benchmark consists of public 2024 yellow taxi trip files published at:
https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page

Raw files are not distributed with this repository. Download the monthly Parquet files separately and prepare them using a common sampling rate:

```bash
mkdir -p data/raw
curl --fail --location --retry 3 \
  'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet' \
  --output data/raw/yellow_tripdata_2024-01.parquet

shiftforge prepare \
  --config configs/experiment.json \
  --input data/raw/yellow_tripdata_2024-01.parquet \
  --output data/processed \
  --sample-bps 1000
```

All eight monthly partitions from January through August are required before running the supervised experiment. The ingestion step produces `data/processed/curated/YYYY-MM.parquet` and `data/processed/manifests/YYYY-MM.json`.

## Model comparison

```bash
shiftforge train \
  --config configs/experiment.json \
  --processed-root data/processed \
  --output artifacts/training \
  --max-train-rows 200000
```

The training command fits an unconditional median predictor, ridge regression and histogram-based gradient boosting. It selects a model using only July and August validation MAE. September is reserved for interval calibration and October through December for final test evaluation. Fixed hyperparameters are used in this milestone, so the current results are an algorithm comparison rather than an optimized benchmark.

Artifacts: `artifacts/training/validation_metrics.json` and `artifacts/training/selected_model.joblib`. The data and model files are excluded from Git. Do not load serialized models from untrusted sources.

## Technical documentation

- [`docs/methodology.md`](docs/methodology.md): experimental partitions and leakage restrictions
- [`docs/ingestion.md`](docs/ingestion.md): reproducible data ingestion and provenance
- [`docs/modeling.md`](docs/modeling.md): estimator design, evaluation metrics and model selection
- [`configs/experiment.json`](configs/experiment.json): data and temporal experiment specification

All project inputs are publicly available or synthetically generated. The implementation requires no proprietary infrastructure or internal datasets.
