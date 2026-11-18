# Public Parquet ingestion protocol

## Source and scope

The benchmark uses publicly available 2024 yellow taxi trip records in monthly Parquet files. Source listing: https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page. Each monthly file is downloaded directly from the public object-storage URL linked on that page. No large benchmark data is redistributed with this repository.

The ingestion engine requires a file named `yellow_tripdata_2024-MM.parquet`. Filenames must match the configured benchmark year. The source URL recorded in the manifest is the canonical public URL derived from the filename. The manifest does not prove the provenance of manually supplied input bytes. Its SHA-256 digest establishes the identity of the actual processed file.

## Dataflow and contract

A lazy Polars scan projects precisely three input fields: `tpep_pickup_datetime`, `tpep_dropoff_datetime` and `PULocationID`. Both timestamps must be timezone-naive datetime columns. The pickup zone must have an integer dtype. Other columns are never read by the query plan.

Rows are classified using first-match rejection precedence:

1. Missing required values
2. Pickup timestamp outside the declared source month
3. Pickup timestamp outside the configured experiment horizon
4. Invalid pickup zone
5. Duration outside the accepted range
6. Label unavailable at the end of the assigned split

Surviving records receive four prediction-time features: pickup zone, local pickup hour, ISO weekday (Monday = 1) and month. The target is elapsed wall-clock minutes between source timestamps. The output also retains pickup time, label-availability time and split identifier for temporal evaluation. Those three fields are metadata or labels, not model predictors.

## Sampling and reproducibility

`sample_bps` controls deterministic sampling in basis points. For example, 1000 selects approximately 10% of accepted records. Selection is based on a seeded Polars hash of the required input fields rather than input row position. This makes sampling independent of record ordering for a fixed Polars implementation. Polars does not promise that hash results remain identical across all future releases. The manifest therefore records the Polars version, random seed and sampling fraction. The output digest is an additional reproducibility check.

All outputs are written outside Git into `data/processed/curated/` and `data/processed/manifests/`. The manifest contains input and output SHA-256 digests, experiment configuration digest, data contract version, version information and rejection counts. Repeated processing of identical inputs returns a cache hit after verifying the output digest. Changes require `--overwrite`.

## Temporal and statistical limitations

The published timestamps are local wall-clock readings without timezone offsets. During daylight saving time transitions, subtraction may not equal physical elapsed time. This limitation remains explicit and will be assessed before fitting final models. The per-split label cutoff excludes trips that finish after their assigned split ends. This may produce a small boundary-selection effect and will be documented in subsequent experiments.

Monthly downloads may contain records with timestamps outside their named month. Such records are rejected by design to avoid duplicate inclusion across monthly partitions. This policy can reduce coverage and is recorded through rejection counts.

The resulting dataset is a reproducible intermediate artifact. It is not yet a fitted model, a drift detector or a completed benchmark.
