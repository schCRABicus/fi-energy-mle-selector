"""Data acquisition layer: shared contracts and per-source clients/fetchers."""

from datasets.api import DataSourceFetcher, FetchRequest
from datasets.energy_charts.client import EnergyChartsClient
from datasets.energy_charts.spot import NordPoolSpotFetcher
from datasets.fingrid.client import FingridClient
from datasets.fingrid.consumption import FingridConsumptionFetcher
from datasets.fingrid.flows import (
    FingridEEFlowFetcher,
    FingridSE1FlowFetcher,
    FingridSE3FlowFetcher,
)
from datasets.fingrid.hydro import FingridHydroFetcher
from datasets.fingrid.nuclear import FingridNuclearFetcher
from datasets.fingrid.wind import FingridWindFetcher
from datasets.fingrid.wind_capacity import FingridWindCapacityFetcher
from datasets.fmi.client import FmiClient
from datasets.fmi.weather import FmiWeatherFetcher

__all__ = [
    "DataSourceFetcher",
    "FetchRequest",
    "EnergyChartsClient",
    "FingridClient",
    "FingridConsumptionFetcher",
    "FingridEEFlowFetcher",
    "FingridHydroFetcher",
    "FingridNuclearFetcher",
    "FingridSE1FlowFetcher",
    "FingridSE3FlowFetcher",
    "FingridWindCapacityFetcher",
    "FingridWindFetcher",
    "FmiClient",
    "FmiWeatherFetcher",
    "NordPoolSpotFetcher",
]
