# Experimental protocol

## Research objective
Compare static, periodically retrained and drift-triggered regressors under temporal distribution shift. The initial benchmark is a public taxi-trip dataset. Synthetic shifts will provide controlled change points for detector evaluation.

## Prediction task
At pickup time, estimate trip duration in minutes. Raw inputs are pickup timestamp and pickup zone. Engineered predictors are pickup zone, hour, weekday and month. The target is computed from drop-off minus pickup timestamp and is unavailable until the trip ends. Outcome-dependent data, including trip distance, destination, fare and payment information, cannot be used as predictors.

## Data contract
Required raw fields are `tpep_pickup_datetime`, `tpep_dropoff_datetime` and `PULocationID`. The initial validity policy accepts pickup zones 1 through 265 and trip durations between 1 and 180 minutes. Invalid records are excluded with explicit rejection reasons and counts. Source timestamps are timezone-naive local clock readings. Daylight saving time ambiguities must be identified and handled before large-scale evaluation.

## Experimental partitions
- Training: January through June 2024
- Validation: July through August 2024
- Calibration: September 2024
- Locked test: October through December 2024

Windows are half-open and assigned by pickup time. Training and calibration inputs must be restricted to examples whose labels were available before the applicable cutoff. Calibration records must not be reused for model selection. Final test results must not influence hyperparameter selection or policy design. Later rolling-origin experiments will respect the same chronological and label-availability constraints.

## Evaluation criteria
Prediction: MAE, RMSE and high-quantile absolute error. Detection: false-alarm rate and delay on synthetic shifts with known change points. Uncertainty: empirical interval coverage and width. Adaptation: predictive error, number of retraining events and computational cost. Results must be obtained from executed experiments, not estimated in advance.

## Source and reproducibility
Public source: https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page

Large input files are retrieved on demand and excluded from Git. Future ingestion manifests will record each source URL, retrieval date, SHA-256 digest, schema version and processing configuration. No raw records, performance measurements or trained models are included at this stage.
