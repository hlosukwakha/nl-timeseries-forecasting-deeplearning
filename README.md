# NL Time Series Forecasting with Deep Learning (Docker + MLflow + Prometheus/Grafana)

Deep learning time series forecasting project using an open Dutch dataset (KNMI hourly station observations) and comparing four modern model families:

1) **Temporal Fusion Transformer (TFT)** – interpretable multi-horizon forecasting
2) **N-BEATS** – neural basis expansion analysis for interpretable forecasting
3) **TimeXer** – Transformer for forecasting with exogenous variables
4) **xLSTMTime** – long-term forecasting architecture built on extended LSTM (xLSTM)

The project is reproducible in Docker and includes experiment tracking (MLflow) plus runtime metrics (Prometheus + Grafana).

## Dataset

We use **KNMI “Uurgegevens van het weer in Nederland”** hourly observations (e.g., station **260 – De Bilt**). KNMI provides zip files per station in 10-year chunks.

> Note on exogenous variables: Weather covariates (wind, pressure, etc.) are *not truly known in the future* in production, but they are available historically for backtesting. This repository supports:
> - **Research/backtesting**: treat observed covariates as “future-known” within the evaluation window (oracle).
> - **Production**: keep only calendar features as future exogenous variables, or swap in real weather forecasts.

## Quick start (Docker)

### 1) Copy env file
```bash
cp.env.example.env
```

### 2) Build and run core services
```bash
docker compose up --build -d mlflow prometheus grafana
```

### 3) Run the end-to-end experiment (downloads data, trains all 4 models, logs to MLflow, writes comparison report)
```bash
docker compose run --rm trainer python -m src.cli run-all

# After completion:
# - data/reports/MODEL_COMPARISON.md
# - data/reports/model_comparison.csv
# - data/reports/*_example.png
```

### 4) Start the inference API (loads the best run and exposes /forecast + /metrics)
```bash
docker compose up --build -d api
```

## What you get

- **MLflow UI**: http://localhost:5000
- **API**: http://localhost:8000
  - `GET /health`
  - `POST /forecast` (JSON body)
  - `GET /metrics` (Prometheus scraping)
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000  (default credentials: `admin` / `admin`)

## Repository layout

```text
.
├── docker-compose.yml
├── docker/
│   ├── Dockerfile.trainer
│   ├── Dockerfile.api
│   └── Dockerfile.mlflow
├── prometheus/
│   └── prometheus.yml
├── grafana/
│   ├── provisioning/
│   │   ├── datasources/datasource.yml
│   │   └── dashboards/dashboard.yml
│   └── dashboards/forecasting-api.json
├── requirements/
│   ├── common.txt
│   ├── train.txt
│   └── serve.txt
└── src/
    ├── cli.py
    ├── settings.py
    ├── data/
    │   ├── knmi_hourly.py
    │   └── features.py
    ├── pipelines/
    │   ├── download.py
    │   ├── preprocess.py
    │   ├── train_all.py
    │   └── evaluate.py
    ├── models/
    │   ├── pt_forecasting.py
    │   └── nf_timexer.py
    └── serve/
        ├── api.py
        └── metrics.py
```

## Models and libraries

- **TFT, N-BEATS, xLSTMTime** via **PyTorch Forecasting**
- **TimeXer** via **NeuralForecast**
- KNMI hourly dataset download links are published on KNMI’s “Uurgegevens” page.

## Configuration

Edit `.env` (or pass env vars) to control station, horizon, and training budget.

Key settings:
- `KNMI_STATION_ID` (default: 260)
- `KNMI_DECADES` (default: `2001-2010,2011-2020,2021-2030`)
- `HORIZON_HOURS` (default: 24)
- `ENCODER_HOURS` (default: 168)
- `EPOCHS` (default: 5)
- `MLFLOW_TRACKING_URI` (default: `http://mlflow:5000` inside Docker)

## Reproducibility & observability

- **MLflow** tracks parameters, metrics, and artifacts (plots + serialized model bundle).
- **Prometheus + Grafana** track API latency and request volume (and you can extend with custom business SLOs).

## Next extensions

- Replace “oracle” weather covariates with KNMI forecast datasets via the KNMI Data Platform Open Data API (anonymous key available).
- Add rolling-origin cross validation (e.g., weekly cutoffs)
- Add conformal prediction intervals and drift monitoring

## License

MIT. See `LICENSE`.
