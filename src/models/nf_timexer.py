from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from neuralforecast import NeuralForecast
from neuralforecast.models import TimeXer

from src.settings import settings


@dataclass
class NFFitResult:
    model_name: str
    metrics: dict


def train_timexer(df: pd.DataFrame, out_dir: Path) -> tuple[NeuralForecast, NFFitResult]:
    # NeuralForecast expects: unique_id, ds, y, and exog columns.
    df = df.copy()
    df["ds"] = pd.to_datetime(df["ds"])

    exog_cols = [c for c in df.columns if c not in {"unique_id", "ds", "y"}]
    # In practice, you should only include future-known features in futr_exog_list.
    # Here we include all covariates for backtesting, including observed weather variables.
    models = [
        TimeXer(
            h=settings.horizon_hours,
            input_size=settings.encoder_hours,
            n_series=1,
            futr_exog_list=exog_cols,
            max_steps=settings.epochs * 200,  # rough budget
            enable_progress_bar=False,
            logger=False,
        )
    ]
    nf = NeuralForecast(models=models, freq=settings.freq)
    nf.fit(df=df)
    return nf, NFFitResult(model_name="timexer", metrics={})
