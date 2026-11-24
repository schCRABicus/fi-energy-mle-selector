"""Data acquisition layer: shared contracts and per-source fetchers."""

from datasets.api import DataSourceFetcher, FetchRequest
from datasets.fingrid_wind.fingrid_wind_fetcher import FingridWindFetcher, run_fetcher

__all__ = [
    "DataSourceFetcher",
    "FetchRequest",
    "FingridWindFetcher",
    "run_fetcher",
]
