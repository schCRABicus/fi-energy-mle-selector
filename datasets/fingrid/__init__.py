"""Fingrid Open Data client and fetchers."""

from datasets.fingrid.client import FingridClient
from datasets.fingrid.wind import FingridWindFetcher
from datasets.fingrid.wind_capacity import FingridWindCapacityFetcher

__all__ = ["FingridClient", "FingridWindCapacityFetcher", "FingridWindFetcher"]