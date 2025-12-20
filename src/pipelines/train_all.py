from __future__ import annotations

from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

from mlflow.tracking import MlflowClient

from src.models.pt_forecasting import _make_pt_dataset, train_tft, train_nbeats, train_xlstmtime
from src.models.nf_timexer import train_timexer
from src.settings import settings



warnings.filterwarnings(
    "ignore",
    message=r"Attribute 'loss' is an instance of `nn\.Module` and is already saved during checkpointing\.",
)
warnings.filterwarnings(
    "ignore",
    message=r"Attribute 'logging_metrics' is an instance of `nn\.Module` and is already saved during checkpointing\.",
)

def _load_dataset() -> pd.DataFrame:
    path = Path(settings.data_dir) / "processed" / f"knmi_station_{settings.knmi_station_id}_hourly.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing processed dataset at {path}. Run preprocess first.")
    return pd.read_parquet(path)


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.nanmean(np.abs(y_true - y_pred)))


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.nanmean((y_true - y_pred) ** 2)))


def _log_example_plot(ds: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray, out_path: Path, title: str) -> None:
    # Plot first forecast window (h points)
    h = settings.horizon_hours
    t = ds.tail(h)["ds"].to_numpy()
    fig = plt.figure()
    plt.plot(t, y_true[:h], label="y")
    plt.plot(t, y_pred[:h], label="yhat")
    plt.title(title)
    plt.legend()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def train_all_models(model_list: list[str]) -> None:
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)

    out_models_dir = Path(settings.model_registry_path)
    out_models_dir.mkdir(parents=True, exist_ok=True)

    df = _load_dataset()

    # Prepare PyTorch Forecasting datasets once (reused across models)
    training, validation, test = _make_pt_dataset(df)
    test_loader = test.to_dataloader(train=False, batch_size=64, num_workers=0, drop_last=True)

    for model_name in model_list:
        with mlflow.start_run(run_name=model_name):
            mlflow.log_params(
                {
                    "station_id": settings.knmi_station_id,
                    "horizon_hours": settings.horizon_hours,
                    "encoder_hours": settings.encoder_hours,
                    "n_windows": settings.n_windows,
                    "epochs": settings.epochs,
                    "batch_size": settings.batch_size,
                    "model": model_name,
                }
            )

            reports_dir = Path(settings.data_dir) / "reports"
            reports_dir.mkdir(parents=True, exist_ok=True)
            fig_path = reports_dir / f"{model_name}_example.png"

            if model_name in {"tft", "nbeats", "xlstmtime"}:
                if model_name == "tft":
                    fit = train_tft(df, out_models_dir)
                elif model_name == "nbeats":
                    fit = train_nbeats(df, out_models_dir)
                else:
                    fit = train_xlstmtime(df, out_models_dir)

                # Reload model from checkpoint for prediction
                ckpt = fit.checkpoint_path
                if not ckpt:
                    raise RuntimeError(f"No checkpoint saved for {model_name}.")

                # Load correct class
                if model_name == "tft":
                    from pytorch_forecasting.models import TemporalFusionTransformer as Model
                elif model_name == "nbeats":
                    from pytorch_forecasting.models import NBeats as Model
                else:
                    from pytorch_forecasting.models import xLSTMTime as Model

                net = Model.load_from_checkpoint(ckpt)
                pred = net.predict(test_loader, return_y=True, mode="prediction", trainer_kwargs={"accelerator": "cpu"})
                # pred is a Prediction object with .output and .y (per BaseModel docs)
                yhat = pred.output.detach().cpu().numpy().reshape(-1)
                y = pred.y.detach().cpu().numpy().reshape(-1)

                test_mae = _mae(y, yhat)
                test_rmse = _rmse(y, yhat)
                mlflow.log_metrics({"test_mae": test_mae, "test_rmse": test_rmse, **fit.metrics})

                # Example plot: last horizon of the dataset
                ds_tail = df.sort_values("ds").tail(settings.horizon_hours).copy()
                _log_example_plot(ds_tail, y_true=y[: settings.horizon_hours], y_pred=yhat[: settings.horizon_hours],
                                  out_path=fig_path, title=f"{model_name} – example window")

                mlflow.log_artifact(str(fig_path), artifact_path="plots")
                mlflow.log_artifact(ckpt, artifact_path="checkpoints")

            elif model_name == "timexer":
                # TimeXer evaluation via NeuralForecast cross_validation on last n_windows
                from neuralforecast import NeuralForecast
                from neuralforecast.models import TimeXer

                df_nf = df.copy()
                df_nf["ds"] = pd.to_datetime(df_nf["ds"])
                exog_cols = [c for c in df_nf.columns if c not in {"unique_id", "ds", "y"}]

                nf = NeuralForecast(
                    models=[
                        TimeXer(
                            h=settings.horizon_hours,
                            input_size=settings.encoder_hours,
                            n_series=1,
                            futr_exog_list=exog_cols,
                            max_steps=settings.epochs * 200,
                            enable_progress_bar=False,
                            logger=False,
                        )
                    ],
                    freq=settings.freq,
                )
                cv_df = nf.cross_validation(
                    df_nf,
                    n_windows=settings.n_windows,
                    step_size=settings.horizon_hours,
                    refit=False,
                    verbose=0,
                )
                # cv_df contains columns: unique_id, ds, cutoff, TimeXer, y
                y = cv_df["y"].to_numpy()
                yhat = cv_df["TimeXer"].to_numpy()

                test_mae = _mae(y, yhat)
                test_rmse = _rmse(y, yhat)
                mlflow.log_metrics({"test_mae": test_mae, "test_rmse": test_rmse})

                # plot last window
                tail = cv_df.tail(settings.horizon_hours)
                fig = plt.figure()
                plt.plot(tail["ds"], tail["y"], label="y")
                plt.plot(tail["ds"], tail["TimeXer"], label="yhat")
                plt.title("timexer – last window")
                plt.legend()
                fig.savefig(fig_path, bbox_inches="tight")
                plt.close(fig)

                mlflow.log_artifact(str(fig_path), artifact_path="plots")

                # Save model (best-effort; TimeXer load issues exist in some versions)
                save_dir = out_models_dir / "timexer"
                save_dir.mkdir(parents=True, exist_ok=True)
                try:
                    nf.save(path=str(save_dir), overwrite=True, save_dataset=False)
                    mlflow.log_artifacts(str(save_dir), artifact_path="timexer_bundle")
                except Exception as e:
                    mlflow.log_text(str(e), artifact_file="timexer_save_error.txt")

            else:
                raise ValueError(f"Unknown model: {model_name}")
