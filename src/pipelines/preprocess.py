from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from src.data.knmi_hourly import read_zips_to_df
from src.data.features import build_supervised_frame
from src.settings import settings


def preprocess_knmi_hourly() -> str:
    raw_dir = Path(settings.data_dir) / "raw" / "knmi"
    zip_paths = sorted(raw_dir.glob(f"uurgeg_{settings.knmi_station_id}_*.zip"))
    if not zip_paths:
        raise FileNotFoundError(f"No KNMI zip files found in {raw_dir}. Run download first.")

    df_raw = read_zips_to_df(zip_paths)
    df = build_supervised_frame(df_raw)
        # ---------------------------------------------------------------------
    # Regularize to a complete hourly index (fix missing timesteps)
    # ---------------------------------------------------------------------
    df = df.sort_values("ds").drop_duplicates(subset=["ds"]).reset_index(drop=True)

    # Build complete hourly timeline
    full_index = pd.date_range(df["ds"].min(), df["ds"].max(), freq="H")
    full = pd.DataFrame({"ds": full_index})

    # Left-join onto full timeline to introduce explicit missing rows
    df = full.merge(df, on="ds", how="left").sort_values("ds").reset_index(drop=True)

    # If STN exists, fill it (station id is constant)
    if "STN" in df.columns:
        df["STN"] = df["STN"].fillna(settings.knmi_station_id).astype("Int64")

    # Recreate calendar features for the newly introduced timestamps
    from src.data.features import add_calendar_features
    df = add_calendar_features(df, ts_col="ds")

    # Identify covariates (everything except ds/y/STN)
    covars = [c for c in df.columns if c not in {"ds", "y", "STN"}]

    # Fill covariates (ffill/bfill is appropriate for many weather vars + always OK for calendar features)
    if covars:
        for c in covars:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df[covars] = df[covars].ffill().bfill().fillna(0.0)

    # Fill target y:
    # - time interpolation handles short gaps well for temperature
    # - then ffill/bfill for any leading/trailing gaps
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df = df.set_index("ds")
    df["y"] = df["y"].interpolate(method="time", limit_direction="both")
    df["y"] = df["y"].ffill().bfill()
    df = df.reset_index()

    # Final safety: drop any remaining missing y (should be none)
    df = df.dropna(subset=["y"]).reset_index(drop=True)

    # Keep y as a real float dtype
    df["y"] = df["y"].astype("float32")


    # Guardrails: y should be numeric and non-null
    df = df.dropna(subset=["y"]).reset_index(drop=True)
    df["y"] = df["y"].astype("float32")


    # add unique_id for panel-friendly libraries
    df["unique_id"] = f"KNMI_{settings.knmi_station_id}"
    df = df[["unique_id", "ds", "y"] + [c for c in df.columns if c not in {"unique_id", "ds", "y", "STN"}]]

    out_dir = Path(settings.data_dir) / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"knmi_station_{settings.knmi_station_id}_hourly.parquet"
    df.to_parquet(out_path, index=False)
    return str(out_path)
