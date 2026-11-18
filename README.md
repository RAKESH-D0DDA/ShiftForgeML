# ShiftForgeML

ShiftForgeML is an independent experimental framework for temporal machine learning, distribution shift detection, uncertainty quantification and adaptive retraining.

The first benchmark uses publicly available taxi trip records. The prediction task is trip-duration regression using only information available at pickup time. Synthetic streams with known change points will support controlled evaluations of drift detectors.

## Development status

**Milestone 1: experimental specification and executable data contracts.**

Implemented:

- Versioned experiment configuration with chronological data partitions
- Feature allowlist that excludes post-outcome predictors
- Row-level validity checks and target computation
- Explicit label-availability checks for temporal evaluation
- Standard-library unit tests and continuous integration

Not yet implemented: downloading or processing Parquet data, fitting ML models, drift detection, conformal intervals, automated retraining or experiment tracking.

## Test the contracts

Run from the repository root with Python 3.11 or newer:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests/unit -v
```

## Project structure

- `configs/experiment.json`: data-source metadata, features and temporal split definitions
- `src/shiftforge/contracts.py`: validation and chronological evaluation contracts
- `tests/unit/test_contracts.py`: executable contract tests
- `docs/methodology.md`: prediction task, leakage constraints and experimental methodology

## Data provenance

The benchmark source is publicly accessible at https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page. No dataset is packaged with this repository. Raw downloads and generated artifacts must remain outside version control.
