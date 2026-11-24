"""Fingrid Open Data wind power client and fetcher."""

from datasets.fingrid_wind.fingrid_client import FingridClient
from datasets.fingrid_wind.fingrid_wind_fetcher import FingridWindFetcher

__all__ = ["FingridClient", "FingridWindFetcher"]