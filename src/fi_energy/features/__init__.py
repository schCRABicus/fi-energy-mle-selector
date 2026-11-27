"""Feature engineering: derived model inputs from the aligned hourly dataset."""

from fi_energy.features.build import (
    ALL_FEATURES,
    build_feature_table,
    describe_features,
    feature_names,
)
from fi_energy.features.grid import (
    GRID_RATIO_FEATURES,
    GRID_RATIO_GROUP,
    add_grid_ratio_features,
    grid_ratio_diagnostics,
)
from fi_energy.features.ratios import safe_ratio
from fi_energy.features.spec import FeatureSpec

__all__ = [
    "ALL_FEATURES",
    "FeatureSpec",
    "GRID_RATIO_FEATURES",
    "GRID_RATIO_GROUP",
    "add_grid_ratio_features",
    "build_feature_table",
    "describe_features",
    "feature_names",
    "grid_ratio_diagnostics",
    "safe_ratio",
]