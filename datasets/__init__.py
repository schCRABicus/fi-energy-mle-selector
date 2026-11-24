"""Data acquisition layer: shared contracts and per-source clients/fetchers."""

from datasets.api import DataSourceFetcher, FetchRequest
from datasets.fingrid_wind.fingrid_client import FingridClient
from datasets.fingrid_wind.fingrid_wind_fetcher import FingridWindFetcher

__all__ = [
    "DataSourceFetcher",
    "FetchRequest",
    "FingridClient",
    "FingridWindFetcher",
]