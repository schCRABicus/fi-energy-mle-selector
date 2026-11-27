"""Fingrid Open Data client and fetchers."""

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

__all__ = [
    "FingridClient",
    "FingridConsumptionFetcher",
    "FingridEEFlowFetcher",
    "FingridHydroFetcher",
    "FingridNuclearFetcher",
    "FingridSE1FlowFetcher",
    "FingridSE3FlowFetcher",
    "FingridWindCapacityFetcher",
    "FingridWindFetcher",
]