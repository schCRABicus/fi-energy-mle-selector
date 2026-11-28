"""Concrete feature groups registered in the default catalog."""

from fi_energy.features.groups.calendar import CalendarFeatureGroup
from fi_energy.features.groups.interactions import InteractionFeatureGroup
from fi_energy.features.groups.thermal_demand import ThermalDemandFeatureGroup
from fi_energy.features.groups.wind_ratios import WindRatiosFeatureGroup

__all__ = [
    "CalendarFeatureGroup",
    "InteractionFeatureGroup",
    "ThermalDemandFeatureGroup",
    "WindRatiosFeatureGroup",
]