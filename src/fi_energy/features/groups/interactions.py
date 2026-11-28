"""Interaction features combining wind supply and thermal demand."""

from __future__ import annotations

import pyarrow as pa
import pyarrow.compute as pc

from fi_energy.features.group import FeatureGroup
from fi_energy.features.spec import FeatureSpec
from fi_energy.features.groups.thermal_demand import HEATING_DEGREE_HOURS
from fi_energy.features.groups.wind_ratios import WIND_PENETRATION_RATIO

__all__ = ["InteractionFeatureGroup"]

INTERACTIONS_GROUP = "interactions"
WIND_TEMP_INTERACTION = "wind_temp_interaction"

_FLOAT32 = pa.float32()


class InteractionFeatureGroup(FeatureGroup):
    """
    Computes ``wind_temp_interaction`` from two earlier feature groups.

    The feature multiplies the wind penetration ratio by heating degree-hours,
    so it is high only when it is cold and wind covers little of consumption —
    the scarce hours when the spot price spikes. It must be computed after the
    ``wind_ratios`` and ``thermal_demand`` groups; the catalog enforces that
    order. A null in either input yields a null interaction.
    """

    name = INTERACTIONS_GROUP
    specs = (
        FeatureSpec(
            name=WIND_TEMP_INTERACTION,
            inputs=(WIND_PENETRATION_RATIO, HEATING_DEGREE_HOURS),
            description=(
                "Wind penetration ratio multiplied by heating degree-hours; "
                "high values flag cold hours with little wind, when heating "
                "demand is met by scarce wind generation."
            ),
        ),
    )

    def validate_inputs(self, table: pa.Table) -> None:
        super().validate_inputs(table)
        for name in self.inputs:
            self._require_numeric(table, name)

    def _compute(self, table: pa.Table) -> pa.Table:
        interaction = pc.multiply(
            pc.cast(table[WIND_PENETRATION_RATIO], _FLOAT32),
            pc.cast(table[HEATING_DEGREE_HOURS], _FLOAT32),
        )
        return table.append_column(
            WIND_TEMP_INTERACTION,
            pc.cast(interaction, _FLOAT32),
        )