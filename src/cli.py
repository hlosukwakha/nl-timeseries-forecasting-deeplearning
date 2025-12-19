from __future__ import annotations

import typer
from rich import print

from src.pipelines.download import download_knmi_hourly
from src.pipelines.preprocess import preprocess_knmi_hourly
from src.pipelines.train_all import train_all_models
from src.pipelines.evaluate import evaluate_all_models

app = typer.Typer(help="NL Time Series Forecasting (KNMI) - CLI")

@app.command()
def download_data():
    """Download KNMI hourly weather data into /data/raw"""
    path = download_knmi_hourly()
    print(f"[green]Downloaded data to:[/green] {path}")

@app.command()
def preprocess():
    """Parse and featurize data into /data/processed"""
    path = preprocess_knmi_hourly()
    print(f"[green]Wrote processed dataset:[/green] {path}")

@app.command()
def train(models: str = typer.Option("tft,nbeats,timexer,xlstmtime", help="Comma-separated model list")):
    """Train selected models and log to MLflow."""
    model_list = [m.strip() for m in models.split(",") if m.strip()]
    train_all_models(model_list=model_list)

@app.command()
def evaluate():
    """Evaluate latest trained models and produce comparison artifacts."""
    evaluate_all_models()

@app.command()
def run_all():
    """End-to-end: download -> preprocess -> train -> evaluate."""
    download_knmi_hourly()
    preprocess_knmi_hourly()
    train_all_models(model_list=["tft", "nbeats", "timexer", "xlstmtime"])
    evaluate_all_models()
    print("[bold green]Done.[/bold green] Open MLflow at http://localhost:5000")

if __name__ == "__main__":
    app()
