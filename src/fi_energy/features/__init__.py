"""Feature engineering: derived model inputs from the aligned hourly dataset."""

from fi_energy.features.build import build_feature_table, describe_features
from fi_energy.features.catalog import DEFAULT_CATALOG, FeatureCatalog
from fi_energy.features.group import FeatureGroup
from fi_energy.features.ratios import safe_ratio
from fi_energy.features.rolling import trailing_mean
from fi_energy.features.spec import FeatureSpec
from fi_energy.features.thermal_demand import ThermalDemandFeatureGroup
from fi_energy.features.wind_ratios import WindRatiosFeatureGroup

__all__ = [
    "DEFAULT_CATALOG",
    "FeatureCatalog",
    "FeatureGroup",
    "FeatureSpec",
    "ThermalDemandFeatureGroup",
    "WindRatiosFeatureGroup",
    "build_feature_table",
    "describe_features",
    "safe_ratio",
    "trailing_mean",
]