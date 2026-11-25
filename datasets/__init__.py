"""Data acquisition layer: shared contracts and per-source clients/fetchers."""

from datasets.api import DataSourceFetcher, FetchRequest
from datasets.fingrid.client import FingridClient
from datasets.fingrid.consumption import FingridConsumptionFetcher
from datasets.fingrid.hydro import FingridHydroFetcher
from datasets.fingrid.nuclear import FingridNuclearFetcher
from datasets.fingrid.wind import FingridWindFetcher
from datasets.fingrid.wind_capacity import FingridWindCapacityFetcher

__all__ = [
    "DataSourceFetcher",
    "FetchRequest",
    "FingridClient",
    "FingridConsumptionFetcher",
    "FingridHydroFetcher",
    "FingridNuclearFetcher",
    "FingridWindCapacityFetcher",
    "FingridWindFetcher",
]