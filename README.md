# NL Time Series Forecasting with Deep Learning (KNMI) — TFT vs N-BEATS vs TimeXer vs xLSTMTime

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED)](https://www.docker.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-ee4c2c)](https://pytorch.org/)
[![Lightning](https://img.shields.io/badge/Lightning-PyTorch%20Lightning-792ee5)](https://lightning.ai/)
[![PyTorch%20Forecasting](https://img.shields.io/badge/PyTorch%20Forecasting-TimeSeries-6aa84f)](https://pytorch-forecasting.readthedocs.io/)
[![NeuralForecast](https://img.shields.io/badge/NeuralForecast-TimeXer-111827)](https://nixtla.github.io/neuralforecast/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2)](https://mlflow.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-API-009688)](https://fastapi.tiangolo.com/)
[![Prometheus](https://img.shields.io/badge/Prometheus-Metrics-E6522C)](https://prometheus.io/)
[![Grafana](https://img.shields.io/badge/Grafana-Dashboards-F46800)](https://grafana.com/)

A Docker-first project that downloads **open Dutch KNMI hourly station weather data**, builds a clean hourly time series (with exogenous features),
and compares **four deep-learning forecasting models**:

- **Temporal Fusion Transformer (TFT)** — interpretable attention-based forecasting  
- **N-BEATS** — interpretable basis expansion forecasting  
- **TimeXer** — forecasting with exogenous variables (NeuralForecast)  
- **xLSTMTime** — extended-LSTM architecture for time series

The pipeline trains all four models, logs metrics/artifacts to **MLflow**, and exposes a **FastAPI** endpoint with **Prometheus** metrics and a **Grafana** dashboard.

> Note on ports (macOS): this project uses **MLflow on port 5002** and the **API on port 8002** (host ports).  
> Container ports remain the defaults (MLflow `5000`, API `8000`).


---

## Why this project exists (short blog-style write-up)

Time series forecasting is everywhere—capacity planning, observability baselining, anomaly detection, demand forecasting, and operational risk monitoring.
But forecasting in real systems is rarely “just a univariate sequence”. There are **exogenous signals** (calendar effects, covariates, known-future variables),
and you need **repeatable experiments** to compare model families fairly.

This repository gives you a practical, reproducible workflow:

1. Pull an **open Dutch dataset** (KNMI station hourly observations).
2. Build a clean hourly series (regularized timeline, engineered features).
3. Train four strong deep-learning forecasting approaches (TFT, N-BEATS, TimeXer, xLSTMTime).
4. Track everything with MLflow (params, metrics, artifacts).
5. Serve forecasts behind an API and add visibility (Prometheus + Grafana).

### Where it can be useful

- Data/ML engineering portfolios (end-to-end reproducible pipeline)
- Benchmarking forecasting approaches on a real-world open dataset
- Observability/SRE: baselines for temperature-like signals, proxy patterns, seasonality effects
- Data architecture demos: reproducible experiments + lineage of artifacts + operational visibility


---

## Tech stack (what each part does)

- **Docker Compose**: runs MLflow, trainer, API, Prometheus, Grafana in a consistent environment.
- **Python 3.11**: runtime for data processing + training + serving.
- **KNMI loader**: downloads and parses hourly station data zips into a normalized dataframe.
- **Feature engineering**: builds a supervised learning frame, adds calendar covariates, and ensures strict hourly continuity.
- **PyTorch + Lightning**: training loop and checkpointing.
- **PyTorch Forecasting**: TFT / N-BEATS / xLSTMTime modeling.
- **NeuralForecast**: TimeXer modeling and backtesting.
- **MLflow**: experiment tracking UI, metrics, and artifacts.
- **FastAPI**: inference service.
- **Prometheus + Grafana**: metrics scraping + dashboarding.


---

## Project tree

```text
.
├── docker-compose.yml
├── Makefile
├── .env.example
├── docker/
│   ├── Dockerfile.trainer
│   ├── Dockerfile.api
│   └── Dockerfile.mlflow
├── requirements/
│   ├── train.txt
│   └── api.txt
├── src/
│   ├── cli.py
│   ├── settings.py
│   ├── data/
│   │   ├── knmi_hourly.py
│   │   └── features.py
│   ├── models/
│   │   ├── pt_forecasting.py
│   │   └── nf_timexer.py
│   ├── pipelines/
│   │   ├── download.py
│   │   ├── preprocess.py
│   │   ├── train_all.py
│   │   └── evaluate.py
│   └── serve/
│       └── api.py
├── prometheus/
│   └── prometheus.yml
└── grafana/
    ├── provisioning/
    │   ├── datasources/
    │   └── dashboards/
    └── dashboards/
        └── api-overview.json
```


---

## Getting started

### 1) Clone and run

```bash
git clone https://github.com/hlosukwakha/nl-timeseries-forecasting-deeplearning.git
cd nl-timeseries-forecasting-deeplearning

cp .env.example .env
make dev
make train
make api
```

### 2) Open the UIs

- **MLflow**: http://localhost:5002  
- **API**: http://localhost:8002  
  - health: `GET /health`
  - forecast: `POST /forecast`
  - metrics: `GET /metrics`
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (default: `admin/admin`)

> If you changed ports, update `docker-compose.yml` port mappings accordingly.


---

## How to use the Makefile

Create the Makefile in the project root (same folder as `docker-compose.yml`).

### Common targets

```bash
make help                    # list all targets
make dev                     # start MLflow + Prometheus + Grafana
make train                   # run download -> preprocess -> train -> evaluate
make preprocess              # download + preprocess only
make api                     # start the FastAPI service
make logs                    # tail logs
make clean-data              # delete local processed artifacts
make rebuild-trainer-nocache # rebuild trainer image without cache
make verify-code             # print key function source from inside container
```

If you prefer `just`, you can add an equivalent `justfile`; Make is used by default here because it is available on most systems.


---

## Configuration

### `.env.example` (and `.env`)

Environment variables used by the pipeline. Typical settings:

- `KNMI_STATION_ID` — station id (e.g., 260 for De Bilt)
- `ENCODER_HOURS` — history length (e.g., 168)
- `HORIZON_HOURS` — forecast length (e.g., 24)
- `N_WINDOWS` — number of backtest windows
- `MLFLOW_TRACKING_URI` — usually `http://mlflow:5000` inside Docker
- `MLFLOW_EXPERIMENT_NAME` — experiment name

### `docker-compose.yml`

- Service wiring for:
  - `mlflow` (mapped to host port **5002** on macOS)
  - `trainer`
  - `api` (mapped to host port **8002** on macOS)
  - `prometheus`
  - `grafana`

### `src/settings.py`

Centralized runtime configuration using Pydantic Settings. If you add new env vars, define them here and load them via `settings.<var>`.

### Prometheus / Grafana configs

- `prometheus/prometheus.yml`: scrape configuration (includes API `/metrics`).
- `grafana/provisioning/*`: datasources + dashboards provisioning.
- `grafana/dashboards/api-overview.json`: a starter dashboard for API metrics.


---

## Running the pipeline manually (without Make)

```bash
docker compose up --build -d mlflow prometheus grafana
docker compose run --rm trainer python -m src.cli run-all
docker compose up --build -d api
```

Artifacts written to:

- `data/processed/` — processed parquet dataset
- `data/reports/` — comparison tables + plots
- `mlruns/` — MLflow tracking backend
- `models/` — local model bundles/checkpoints


---

## Outputs and evaluation

After `make train`, you should see:

- `data/reports/MODEL_COMPARISON.md`
- `data/reports/model_comparison.csv`
- `data/reports/*_example.png`

MLflow will contain:
- params (station id, horizon, encoder, model)
- metrics (val loss, test MAE/RMSE where available)
- artifacts (plots, checkpoints)


---

## API usage

Example forecast request:

```bash
# You can use JQ to make the output pretty
curl -X POST "http://localhost:8002/forecast" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tft",
    "horizon_hours": 24,
    "series": [
      {"ds": "2025-12-20T00:00:00", "y": 6.1},
      {"ds": "2025-12-20T01:00:00", "y": 6.0},
      {"ds": "2025-12-20T02:00:00", "y": 5.9}
    ]
  }'
```

Prometheus metrics:

```bash
curl "http://localhost:8002/metrics"
```


---

## Testing

This project is designed primarily as a reproducible experiment runner. If you add tests:

- unit-test parsing (`src/data/knmi_hourly.py`)
- unit-test feature engineering (`src/data/features.py`)
- smoke-test training dataset construction (`src/models/pt_forecasting.py`)

Suggested structure:

```text
tests/
  test_knmi_parser.py
  test_features.py
  test_dataset_build.py
```


---

## Extending the project

Ideas:

1. **Add more stations** as separate `unique_id`s (true panel forecasting).
2. Add stricter evaluation:
   - “future-known only” covariates (calendar)
   - remove “oracle weather” covariates from val/test
3. Add SHAP or built-in interpretability exports for TFT and N-BEATS.
4. Add a scheduled retraining job (Cron + Docker) and publish metrics as time series.
5. Add a “model registry” step (MLflow Model Registry) if you run a backend that supports it.


---

## Common errors and troubleshooting

### Docker image caching (changes not reflected)
If you edit Python code but the container still runs the old version, rebuild without cache:

```bash
make rebuild-trainer-nocache
```

If you mount `./src` into the trainer container, you can iterate without rebuilds (dev-only).

### `Invalid Host header - possible DNS rebinding attack detected` (MLflow 403)
MLflow may reject Docker host headers. Add to `mlflow` service env in `docker-compose.yml`:

- `MLFLOW_SERVER_ALLOWED_HOSTS=localhost,127.0.0.1,mlflow,mlflow:5000`

Or (dev-only) disable the middleware:
- `MLFLOW_SERVER_DISABLE_SECURITY_MIDDLEWARE=true`

### KNMI parsing failures (header not found)
Some station files place the header on a commented line. If parsing fails:
- delete `data/raw/knmi` and re-run the download + preprocess steps.

```bash
make clean-data
make preprocess
```

### `TypeError: float() ... NAType`
Usually indicates missing values sneaking into features/target. This project:
- drops missing targets
- interpolates/fills target
- fills remaining covariates
- enforces float32 features before writing parquet

If you still see it, delete `data/processed` and rerun preprocess.

### `AssertionError: filters should not remove entries ...` (PyTorch Forecasting)
Typically means validation/test were built without encoder history. Fix is in `TimeSeriesDataSet.from_dataset(...)` construction in `src/models/pt_forecasting.py`.

### Performance warnings about `num_workers`
Lightning may warn about `num_workers=0`. You can raise to `2` for speed; on macOS Docker, very high worker counts can cause issues.


---

## Scripts, queries, and YAML files

### Key scripts/modules

- `src/cli.py`  
  Entrypoint with commands like `run-all`, `download-data`, `preprocess`, `train`, `evaluate`.

- `src/pipelines/download.py`  
  Downloads KNMI hourly zips per time chunk.

- `src/pipelines/preprocess.py`  
  Parses + feature engineers + regularizes to a complete hourly time index; writes parquet.

- `src/pipelines/train_all.py`  
  Trains all 4 models, logs to MLflow, writes plots and evaluation artifacts.

- `src/pipelines/evaluate.py`  
  Reads latest MLflow runs and produces `MODEL_COMPARISON.md` and CSV.

- `src/models/pt_forecasting.py`  
  TFT, N-BEATS, xLSTMTime training + correct val/test dataset construction.

- `src/models/nf_timexer.py`  
  TimeXer training/evaluation via NeuralForecast.

- `src/serve/api.py`  
  FastAPI inference service.

### YAML files (infra/config)

- `docker-compose.yml`  
  Service wiring, ports, volumes, env vars.

- `prometheus/prometheus.yml`  
  Scrape config for the API metrics endpoint.

Grafana provisioning YAMLs (in `grafana/provisioning/`) define:
- which data source Grafana should use (Prometheus)
- which dashboards to load automatically


---

## Notes on ports (macOS)

You requested:

- MLflow host port: **5002**
- API host port: **8002**

Ensure `docker-compose.yml` contains mappings like:

```yaml
mlflow:
  ports:
    - "5002:5000"
api:
  ports:
    - "8002:8000"
```


---

## Signature

Built by **@hlosukwakha**
