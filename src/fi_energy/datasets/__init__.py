"""Data acquisition layer: shared contracts and per-source clients/fetchers."""

from fi_energy.datasets.api import DataSourceFetcher, FetchRequest
from fi_energy.datasets.energy_charts.client import EnergyChartsClient
from fi_energy.datasets.energy_charts.spot import NordPoolSpotFetcher
from fi_energy.datasets.fingrid.client import FingridClient
from fi_energy.datasets.fingrid.consumption import FingridConsumptionFetcher
from fi_energy.datasets.fingrid.flows import (
    FingridEEFlowFetcher,
    FingridSE1FlowFetcher,
    FingridSE3FlowFetcher,
)
from fi_energy.datasets.fingrid.hydro import FingridHydroFetcher
from fi_energy.datasets.fingrid.nuclear import FingridNuclearFetcher
from fi_energy.datasets.fingrid.wind import FingridWindFetcher
from fi_energy.datasets.fingrid.wind_capacity import FingridWindCapacityFetcher
from fi_energy.datasets.fmi.client import FmiClient
from fi_energy.datasets.fmi.weather import FmiWeatherFetcher

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
