from __future__ import annotations

from pathlib import Path
import warnings

import mlflow
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from src.models.pt_forecasting import _make_pt_dataset, train_tft, train_nbeats, train_xlstmtime
from src.settings import settings
from neuralforecast import NeuralForecast
from neuralforecast.models import TimeXer


warnings.filterwarnings(
    "ignore",
    message=r"Attribute 'loss' is an instance of `nn\.Module` and is already saved during checkpointing\.",
)
warnings.filterwarnings(
    "ignore",
    message=r"Attribute 'logging_metrics' is an instance of `nn\.Module` and is already saved during checkpointing\.",
)


def _to_numpy_1d(x) -> np.ndarray:
    while isinstance(x, (tuple, list)) and len(x) > 0:
        x = x[0]

    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    else:
        x = np.asarray(x)

    return x.reshape(-1)


def _load_dataset() -> pd.DataFrame:
    path = Path(settings.data_dir) / "processed" / f"knmi_station_{settings.knmi_station_id}_hourly.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing processed dataset at {path}. Run preprocess first.")
    return pd.read_parquet(path)


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.nanmean(np.abs(y_true - y_pred)))


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.nanmean((y_true - y_pred) ** 2)))


def _log_example_plot(
    ds_tail: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray, out_path: Path, title: str
) -> None:
    fig = plt.figure()
    plt.plot(ds_tail["ds"].to_numpy(), y_true, label="y")
    plt.plot(ds_tail["ds"].to_numpy(), y_pred, label="yhat")
    plt.title(title)
    plt.legend()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def train_all_models(model_list: list[str]) -> None:
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)

    out_models_dir = Path(settings.model_registry_path)
    out_models_dir.mkdir(parents=True, exist_ok=True)

    reports_dir = Path(settings.data_dir) / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    df = _load_dataset().sort_values("ds").reset_index(drop=True)

    # Prepare datasets once (evaluation uses 'test')
    _, _, test = _make_pt_dataset(df)

    num_workers = int(getattr(settings, "num_workers", 2))
    test_loader = test.to_dataloader(
        train=False,
        batch_size=64,
        num_workers=num_workers,
        drop_last=False,
    )

    for model_name in model_list:
        fig_path = reports_dir / f"{model_name}_example.png"

        with mlflow.start_run(run_name=model_name):
            mlflow.log_params(
                {
                    "station_id": settings.knmi_station_id,
                    "horizon_hours": settings.horizon_hours,
                    "encoder_hours": settings.encoder_hours,
                    "n_windows": settings.n_windows,
                    "epochs": settings.epochs,
                    "batch_size": settings.batch_size,
                    "num_workers": num_workers,
                    "model": model_name,
                }
            )

            # ---------------------------------------------------------
            # PyTorch Forecasting models
            # ---------------------------------------------------------
            if model_name in {"tft", "nbeats", "xlstmtime"}:
                if model_name == "tft":
                    fit = train_tft(df, out_models_dir)
                    from pytorch_forecasting.models import TemporalFusionTransformer as Model
                elif model_name == "nbeats":
                    fit = train_nbeats(df, out_models_dir)
                    from pytorch_forecasting.models import NBeats as Model
                else:
                    fit = train_xlstmtime(df, out_models_dir)
                    from pytorch_forecasting.models import xLSTMTime as Model

                ckpt = fit.checkpoint_path
                if not ckpt:
                    raise RuntimeError(f"No checkpoint saved for {model_name}.")

                net = Model.load_from_checkpoint(ckpt)

                pred = net.predict(
                    test_loader,
                    return_y=True,
                    mode="prediction",
                    trainer_kwargs={"accelerator": "cpu"},
                )

                if hasattr(pred, "output") and hasattr(pred, "y"):
                    yhat = _to_numpy_1d(pred.output)
                    y = _to_numpy_1d(pred.y)
                else:
                    yhat = _to_numpy_1d(pred)
                    y = np.full_like(yhat, np.nan, dtype=float)

                test_mae = _mae(y, yhat)
                test_rmse = _rmse(y, yhat)

                mlflow.log_metrics({"test_mae": test_mae, "test_rmse": test_rmse, **fit.metrics})

                # Example plot: align timestamps (last h) with last h predictions
                h = int(settings.horizon_hours)
                ds_tail = df.tail(h).copy()
                y_tail = y[-h:] if len(y) >= h else y
                yhat_tail = yhat[-h:] if len(yhat) >= h else yhat

                _log_example_plot(
                    ds_tail=ds_tail,
                    y_true=y_tail,
                    y_pred=yhat_tail,
                    out_path=fig_path,
                    title=f"{model_name} – example window",
                )

                mlflow.log_artifact(str(fig_path), artifact_path="plots")
                mlflow.log_artifact(ckpt, artifact_path="checkpoints")

            # ---------------------------------------------------------
            # NeuralForecast TimeXer
            # ---------------------------------------------------------
            elif model_name == "timexer":
                df_nf = df.copy()
                df_nf["ds"] = pd.to_datetime(df_nf["ds"])
                exog_cols = [c for c in df_nf.columns if c not in {"unique_id", "ds", "y"}]

                model = TimeXer(
                    h=int(settings.horizon_hours),
                    input_size=int(settings.encoder_hours),
                    n_series=1,
                    futr_exog_list=exog_cols,
                    max_steps=int(settings.epochs) * 200,
                    enable_progress_bar=False,
                    logger=False,
                )

                nf = NeuralForecast(models=[model], freq=settings.freq)

                cv_df = nf.cross_validation(
                    df=df_nf,
                    n_windows=int(settings.n_windows),
                    step_size=int(settings.horizon_hours),
                    refit=False,
                    verbose=False,
                )

                y = cv_df["y"].to_numpy()
                yhat = cv_df["TimeXer"].to_numpy()

                test_mae = _mae(y, yhat)
                test_rmse = _rmse(y, yhat)
                mlflow.log_metrics({"test_mae": test_mae, "test_rmse": test_rmse})

                # Example plot: last window
                h = int(settings.horizon_hours)
                tail = cv_df.tail(h)
                fig = plt.figure()
                plt.plot(tail["ds"], tail["y"], label="y")
                plt.plot(tail["ds"], tail["TimeXer"], label="yhat")
                plt.title("timexer – last window")
                plt.legend()
                fig.savefig(fig_path, bbox_inches="tight")
                plt.close(fig)

                mlflow.log_artifact(str(fig_path), artifact_path="plots")

                # Save bundle (best-effort)
                save_dir = out_models_dir / "timexer"
                save_dir.mkdir(parents=True, exist_ok=True)
                try:
                    nf.save(path=str(save_dir), overwrite=True, save_dataset=False)
                    mlflow.log_artifacts(str(save_dir), artifact_path="timexer_bundle")
                except Exception as e:
                    mlflow.log_text(str(e), artifact_file="timexer_save_error.txt")

            else:
                raise ValueError(f"Unknown model: {model_name}")
