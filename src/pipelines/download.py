from __future__ import annotations

from pathlib import Path

from src.data.knmi_hourly import KNMISource, download_sources
from src.settings import settings


def download_knmi_hourly() -> str:
    raw_dir = Path(settings.data_dir) / "raw" / "knmi"
    decades = [d.strip() for d in settings.knmi_decades.split(",") if d.strip()]
    sources = [KNMISource(station_id=settings.knmi_station_id, decade=d) for d in decades]
    paths = download_sources(sources, out_dir=raw_dir)
    return str(raw_dir)
