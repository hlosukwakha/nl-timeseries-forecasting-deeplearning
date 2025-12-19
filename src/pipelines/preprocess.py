from __future__ import annotations

from pathlib import Path
import pandas as pd

from src.data.knmi_hourly import read_zips_to_df
from src.data.features import build_supervised_frame, add_calendar_features
from src.settings import settings


def preprocess_knmi_hourly() -> str:
    raw_dir = Path(settings.data_dir) / "raw" / "knmi"
    zip_paths = sorted(raw_dir.glob(f"uurgeg_{settings.knmi_station_id}_*.zip"))
    if not zip_paths:
        raise FileNotFoundError(f"No KNMI zip files found in {raw_dir}. Run download first.")

    # Base frame (ds, y, covariates)
    df_raw = read_zips_to_df(zip_paths)
    df = build_supervised_frame(df_raw)

    # ---------------------------------------------------------------------
    # Regularize to a complete hourly index (fix missing timesteps)
    # ---------------------------------------------------------------------
    df = df.sort_values("ds").drop_duplicates(subset=["ds"]).reset_index(drop=True)

    full_index = pd.date_range(df["ds"].min(), df["ds"].max(), freq="h")
    full = pd.DataFrame({"ds": full_index})

    # Introduce missing rows explicitly
    df = full.merge(df, on="ds", how="left").sort_values("ds").reset_index(drop=True)

    # Recompute calendar features for the full timeline
    df = add_calendar_features(df, ts_col="ds")

    # Identify covariates (everything except ds/y/STN)
    covars = [c for c in df.columns if c not in {"ds", "y", "STN"}]

    # Fill covariates
    if covars:
        for c in covars:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df[covars] = df[covars].ffill().bfill().fillna(0.0)

    # Fill target y
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df = df.set_index("ds")
    df["y"] = df["y"].interpolate(method="time", limit_direction="both")
    df["y"] = df["y"].ffill().bfill()
    df = df.reset_index()

    # Final safety
    df = df.dropna(subset=["y"]).reset_index(drop=True)
    df["y"] = df["y"].astype("float32")

    # Add group key for panel-friendly libs
    df["unique_id"] = f"KNMI_{settings.knmi_station_id}"

    # Keep only model-relevant columns (exclude STN if present)
    df = df[["unique_id", "ds", "y"] + [c for c in df.columns if c not in {"unique_id", "ds", "y", "STN"}]]

    # Enforce numeric + non-null inputs (PyTorch Forecasting strictness)
    feature_cols = [c for c in df.columns if c not in {"unique_id", "ds"}]
    for c in feature_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df[feature_cols] = df[feature_cols].fillna(0.0).astype("float32")

    out_dir = Path(settings.data_dir) / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"knmi_station_{settings.knmi_station_id}_hourly.parquet"
    df.to_parquet(out_path, index=False)

    return str(out_path)
