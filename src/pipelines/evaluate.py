from __future__ import annotations

from pathlib import Path

import pandas as pd
from mlflow.tracking import MlflowClient

from src.settings import settings


def evaluate_all_models() -> None:
    client = MlflowClient(tracking_uri=settings.mlflow_tracking_uri)

    exp = client.get_experiment_by_name(settings.mlflow_experiment_name)
    if exp is None:
        raise RuntimeError(f"MLflow experiment not found: {settings.mlflow_experiment_name}")

    # Pull latest run per model name (by start_time)
    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=["attributes.start_time DESC"],
        max_results=200,
    )

    latest = {}
    for r in runs:
        name = r.data.params.get("model")
        if not name:
            continue
        if name not in {"tft", "nbeats", "timexer", "xlstmtime"}:
            continue
        if name not in latest:
            latest[name] = r

    rows = []
    for name in ["tft", "nbeats", "timexer", "xlstmtime"]:
        r = latest.get(name)
        if r is None:
            rows.append({"model": name, "run_id": None, "test_mae": None, "test_rmse": None})
            continue
        rows.append(
            {
                "model": name,
                "run_id": r.info.run_id,
                "test_mae": r.data.metrics.get("test_mae"),
                "test_rmse": r.data.metrics.get("test_rmse"),
                "val_loss": r.data.metrics.get("val_loss"),
            }
        )

    out_dir = Path(settings.data_dir) / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "model_comparison.csv", index=False)

    # Markdown report
    md = ["# Model comparison (latest runs)", "", df.to_markdown(index=False), "", "Open MLflow: http://localhost:5000"]
    (out_dir / "MODEL_COMPARISON.md").write_text("\n".join(md), encoding="utf-8")
