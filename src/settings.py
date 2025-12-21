from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore", protected_namespaces=())

    # Data
    knmi_station_id: int = 260
    knmi_decades: str = "2001-2010,2011-2020,2021-2030"
    data_dir: str = "/data"

    # Forecasting
    horizon_hours: int = 24
    encoder_hours: int = 168
    freq: str = "H"

    # Evaluation
    n_windows: int = 7  # backtest windows for metrics

    # Training
    epochs: int = 5
    batch_size: int = 256
    seed: int = 42

    # Tracking
    mlflow_tracking_uri: str = "http://mlflow:5000"
    mlflow_experiment_name: str = "knmi_hourly_forecasting"
    model_registry_path: str = "/models"

    # API
    api_port: int = 8000


settings = Settings()
