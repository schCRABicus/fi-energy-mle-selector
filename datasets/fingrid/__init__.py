"""Fingrid Open Data client and fetchers."""

from datasets.fingrid.client import FingridClient
from datasets.fingrid.consumption import FingridConsumptionFetcher
from datasets.fingrid.hydro import FingridHydroFetcher
from datasets.fingrid.nuclear import FingridNuclearFetcher
from datasets.fingrid.wind import FingridWindFetcher
from datasets.fingrid.wind_capacity import FingridWindCapacityFetcher

__all__ = [
    "FingridClient",
    "FingridConsumptionFetcher",
    "FingridHydroFetcher",
    "FingridNuclearFetcher",
    "FingridWindCapacityFetcher",
    "FingridWindFetcher",
]