from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple
import shutil

import numpy as np
import pandas as pd
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import MAE
from pytorch_forecasting.models import NBeats, TemporalFusionTransformer, xLSTMTime

from src.settings import settings


# ---------------------------------------------------------------------
# Public result type
# ---------------------------------------------------------------------
@dataclass
class PTFitResult:
    model_name: str
    checkpoint_path: str
    metrics: dict


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def _ensure_expected_columns(df: pd.DataFrame) -> None:
    required = {"unique_id", "ds", "y"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in processed dataset: {missing}")


def _make_time_idx(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data["ds"] = pd.to_datetime(data["ds"], errors="coerce")
    data = data.dropna(subset=["ds"]).sort_values(["unique_id", "ds"]).reset_index(drop=True)

    # Hourly integer index
    data["time_idx"] = ((data["ds"] - data["ds"].min()).dt.total_seconds() // 3600).astype(int)
    return data


def _to_numpy_1d(x) -> np.ndarray:
    """
    Convert pytorch-forecasting outputs to 1D numpy arrays.

    Some versions return tuples/lists (tensor, metadata) or nested containers.
    We unwrap until we hit a tensor/array.
    """
    while isinstance(x, (tuple, list)) and len(x) > 0:
        x = x[0]

    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().reshape(-1)

    return np.asarray(x).reshape(-1)


# ---------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------
def _make_pt_dataset(
    df: pd.DataFrame,
    use_exog: bool = True,
) -> Tuple[TimeSeriesDataSet, TimeSeriesDataSet, TimeSeriesDataSet]:
    """
    Build training/validation/test datasets.

    use_exog=True  -> include engineered covariates as known reals (for TFT)
    use_exog=False -> target-only dataset (for univariate-only models like N-BEATS; safe for xLSTMTime)
    """
    _ensure_expected_columns(df)
    data = _make_time_idx(df)

    if use_exog:
        covars = [c for c in data.columns if c not in {"unique_id", "ds", "y", "time_idx"}]
        known_reals = covars
    else:
        # Keep only required cols; no exogenous variables
        data = data[["unique_id", "ds", "y", "time_idx"]].copy()
        known_reals = []

    # For univariate-only models (N-BEATS / sometimes xLSTMTime), we must ensure the ONLY input is y.
    # add_target_scales/add_encoder_length add extra inputs, so disable them when use_exog=False.
    add_target_scales = True if use_exog else False
    add_encoder_length = True if use_exog else False

    common_kwargs = dict(
        time_idx="time_idx",
        target="y",
        group_ids=["unique_id"],
        max_encoder_length=int(settings.encoder_hours),
        max_prediction_length=int(settings.horizon_hours),
        time_varying_known_reals=known_reals,
        time_varying_unknown_reals=["y"],
        target_normalizer=GroupNormalizer(groups=["unique_id"]),
        add_relative_time_idx=False,
        add_target_scales=add_target_scales,
        add_encoder_length=add_encoder_length,
        allow_missing_timesteps=True,
    )


    max_time = int(data["time_idx"].max())
    n_windows = int(getattr(settings, "n_windows", 7))
    holdout = int(settings.horizon_hours) * n_windows

    test_cut = max_time - holdout
    val_cut = test_cut - holdout

    # Clamp cutoffs to ensure we have at least one full encoder+decoder sequence
    min_required_end = int(settings.encoder_hours + settings.horizon_hours)
    if val_cut < min_required_end:
        val_cut = min_required_end
    if test_cut <= val_cut + int(settings.horizon_hours):
        test_cut = val_cut + int(settings.horizon_hours)
    if test_cut >= max_time:
        test_cut = max_time - int(settings.horizon_hours)

    training = TimeSeriesDataSet(data[data.time_idx <= val_cut], **common_kwargs)

    validation_data = data[data.time_idx <= test_cut]
    validation = TimeSeriesDataSet.from_dataset(
        training,
        validation_data,
        min_prediction_idx=val_cut + 1,
        stop_randomization=True,
    )

    test = TimeSeriesDataSet.from_dataset(
        training,
        data,
        min_prediction_idx=test_cut + 1,
        stop_randomization=True,
    )

    if len(training) == 0:
        raise RuntimeError(
            f"Training dataset empty. encoder={settings.encoder_hours} horizon={settings.horizon_hours} "
            f"val_cut={val_cut} max_time={max_time} use_exog={use_exog}"
        )
    if len(validation) == 0:
        raise RuntimeError(
            f"Validation dataset empty. encoder={settings.encoder_hours} horizon={settings.horizon_hours} "
            f"val_cut={val_cut} test_cut={test_cut} max_time={max_time} use_exog={use_exog}"
        )
    if len(test) == 0:
        raise RuntimeError(
            f"Test dataset empty. encoder={settings.encoder_hours} horizon={settings.horizon_hours} "
            f"test_cut={test_cut} max_time={max_time} use_exog={use_exog}"
        )

    return training, validation, test


# ---------------------------------------------------------------------
# Trainer + checkpoint helpers
# ---------------------------------------------------------------------
def _trainer(out_dir: Path, name: str) -> Trainer:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Avoid Lightning warning: directory exists and not empty
    ckpt_dir = out_dir / "checkpoints"
    if ckpt_dir.exists():
        shutil.rmtree(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_cb = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename=f"{name}" + "-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
    )

    return Trainer(
        max_epochs=int(settings.epochs),
        accelerator="cpu",
        enable_checkpointing=True,
        default_root_dir=str(out_dir),
        callbacks=[
            checkpoint_cb,
            EarlyStopping(monitor="val_loss", patience=3, mode="min"),
        ],
        logger=CSVLogger(save_dir=str(out_dir.parent), name=f"{name}_logs"),
        enable_model_summary=False,
    )


def _checkpoint_path_or_raise(trainer: Trainer, model_name: str) -> str:
    ckpt_cb = next((c for c in trainer.callbacks if isinstance(c, ModelCheckpoint)), None)
    if ckpt_cb is None:
        raise RuntimeError(f"No ModelCheckpoint callback found for {model_name}.")

    ckpt_path = ckpt_cb.best_model_path or ckpt_cb.last_model_path
    if not ckpt_path:
        metric_keys = list(trainer.callback_metrics.keys())
        raise RuntimeError(
            f"No checkpoint saved for {model_name}. "
            f"Make sure 'val_loss' is logged. callback_metrics keys={metric_keys}"
        )
    return ckpt_path


# ---------------------------------------------------------------------
# Train functions
# ---------------------------------------------------------------------
def train_tft(df: pd.DataFrame, out_dir: Path) -> PTFitResult:
    training, validation, _ = _make_pt_dataset(df, use_exog=True)

    num_workers = int(getattr(settings, "num_workers", 2))
    train_loader = training.to_dataloader(train=True, batch_size=int(settings.batch_size), num_workers=num_workers)
    val_loader = validation.to_dataloader(train=False, batch_size=int(settings.batch_size), num_workers=num_workers)

    model = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=3e-3,
        hidden_size=32,
        attention_head_size=4,
        dropout=0.1,
        hidden_continuous_size=16,
        loss=MAE(),
        log_interval=10,
        reduce_on_plateau_patience=2,
    )

    run_dir = out_dir / "tft"
    trainer = _trainer(run_dir, "tft")
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    ckpt_path = _checkpoint_path_or_raise(trainer, "tft")
    return PTFitResult(
        model_name="tft",
        checkpoint_path=ckpt_path,
        metrics={"val_loss": float(trainer.callback_metrics.get("val_loss", float("nan")))},
    )


def train_nbeats(df: pd.DataFrame, out_dir: Path) -> PTFitResult:
    training, validation, _ = _make_pt_dataset(df, use_exog=False)

    num_workers = int(getattr(settings, "num_workers", 2))
    train_loader = training.to_dataloader(train=True, batch_size=int(settings.batch_size), num_workers=num_workers)
    val_loader = validation.to_dataloader(train=False, batch_size=int(settings.batch_size), num_workers=num_workers)

    model = NBeats.from_dataset(
        training,
        learning_rate=3e-3,
        log_interval=10,
        log_val_interval=1,
        weight_decay=1e-2,
        widths=[256, 256],
        backcast_loss_ratio=0.0,
        loss=MAE(),
    )

    run_dir = out_dir / "nbeats"
    trainer = _trainer(run_dir, "nbeats")
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    ckpt_path = _checkpoint_path_or_raise(trainer, "nbeats")
    return PTFitResult(
        model_name="nbeats",
        checkpoint_path=ckpt_path,
        metrics={"val_loss": float(trainer.callback_metrics.get("val_loss", float("nan")))},
    )


def train_xlstmtime(df: pd.DataFrame, out_dir: Path) -> PTFitResult:
    # xLSTMTime: keep univariate dataset to satisfy "target-only" constraints
    training, validation, _ = _make_pt_dataset(df, use_exog=False)

    num_workers = int(getattr(settings, "num_workers", 2))
    train_loader = training.to_dataloader(train=True, batch_size=int(settings.batch_size), num_workers=num_workers)
    val_loader = validation.to_dataloader(train=False, batch_size=int(settings.batch_size), num_workers=num_workers)

    # Infer input_size from the dataset (encoder_cont shape: [B, encoder_len, n_features])
    probe_loader = training.to_dataloader(train=False, batch_size=2, num_workers=0)
    x_probe, _ = next(iter(probe_loader))
    input_size = int(x_probe["encoder_cont"].shape[-1])

    # output_size for xLSTMTime is the forecast horizon length
    output_size = int(settings.horizon_hours)

    model = xLSTMTime.from_dataset(
        training,
        input_size=input_size,
        output_size=output_size,
        learning_rate=3e-3,
        hidden_size=64,
        xlstm_type="slstm",
        num_layers=1,
        dropout=0.1,
        loss=MAE(),
        log_interval=10,
    )

    ckpt_dir = out_dir / "xlstmtime"
    trainer = _trainer(ckpt_dir, "xlstmtime")
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    ckpt_path = _checkpoint_path_or_raise(trainer, "xlstmtime")

    return PTFitResult(
        model_name="xlstmtime",
        checkpoint_path=ckpt_path,
        metrics={"val_loss": float(trainer.callback_metrics.get("val_loss", float("nan")))},
    )



# ---------------------------------------------------------------------
# Prediction / evaluation helper
# ---------------------------------------------------------------------
@torch.no_grad()
def predict_pt_checkpoint(model_name: str, checkpoint_path: str, df: pd.DataFrame) -> pd.DataFrame:
    """
    Load a trained checkpoint and produce predictions on the test split.

    Important: uses the correct dataset schema per model.
      - TFT uses exogenous variables
      - N-BEATS / xLSTMTime are treated as target-only (univariate) to satisfy model constraints
    """
    if model_name == "tft":
        use_exog = True
        Model = TemporalFusionTransformer
    elif model_name == "nbeats":
        use_exog = False
        Model = NBeats
    elif model_name == "xlstmtime":
        use_exog = False
        Model = xLSTMTime
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    _, _, test = _make_pt_dataset(df, use_exog=use_exog)
    test_loader = test.to_dataloader(
        train=False,
        batch_size=64,
        num_workers=int(getattr(settings, "num_workers", 2)),
        drop_last=False,
    )

    net = Model.load_from_checkpoint(checkpoint_path)

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

    return pd.DataFrame({"y": y, "yhat": yhat})
