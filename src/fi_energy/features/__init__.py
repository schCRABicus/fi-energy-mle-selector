"""Feature engineering: derived model inputs from the aligned hourly dataset."""

from fi_energy.features.build import build_feature_table, describe_features
from fi_energy.features.catalog import DEFAULT_CATALOG, FeatureCatalog
from fi_energy.features.group import FeatureGroup
from fi_energy.features.groups import (
    CalendarFeatureGroup,
    InteractionFeatureGroup,
    ThermalDemandFeatureGroup,
    WindRatiosFeatureGroup,
)
from fi_energy.features.ratios import safe_ratio
from fi_energy.features.rolling import trailing_mean
from fi_energy.features.spec import FeatureSpec

__all__ = [
    "DEFAULT_CATALOG",
    "CalendarFeatureGroup",
    "FeatureCatalog",
    "FeatureGroup",
    "FeatureSpec",
    "InteractionFeatureGroup",
    "ThermalDemandFeatureGroup",
    "WindRatiosFeatureGroup",
    "build_feature_table",
    "describe_features",
    "safe_ratio",
    "trailing_mean",
]