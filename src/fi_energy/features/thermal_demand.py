"""Thermal-demand features derived from Helsinki hourly temperature."""

from __future__ import annotations

from typing import Any

import pyarrow as pa
import pyarrow.compute as pc

from fi_energy.features.group import FeatureGroup
from fi_energy.features.rolling import trailing_mean
from fi_energy.features.spec import FeatureSpec

__all__ = ["ThermalDemandFeatureGroup"]

THERMAL_DEMAND_GROUP = "thermal_demand"

TEMPERATURE_COLUMN = "helsinki_temperature_c"
HEATING_BASE_TEMPERATURE_C = 17.0
HEATING_DEGREE_HOURS = "heating_degree_hours"
TEMP_ROLLING_24H = "temp_rolling_24h"
TEMP_ROLLING_72H = "temp_rolling_72h"

_FLOAT32 = pa.float32()
_FLOAT64 = pa.float64()
_ZERO32 = pa.scalar(0.0, type=_FLOAT32)
_ZERO64 = pa.scalar(0.0, type=_FLOAT64)


def heating_degree_hours(
    temperature: pa.Array | pa.ChunkedArray,
    base: float = HEATING_BASE_TEMPERATURE_C,
) -> pa.Array:
    """
    Heating demand in degree-hours: ``max(0, base - temperature)``.

    Each row covers one hour, so a value is also the degree-hours accumulated
    during that hour. Null temperatures stay null.
    """
    delta = pc.subtract(
        pa.scalar(base, type=_FLOAT64),
        pc.cast(temperature, _FLOAT64),
    )
    return pc.cast(
        # skip_nulls=False so a missing temperature is not reported as a
        # mild hour with zero heating demand.
        pc.max_element_wise(delta, _ZERO64, skip_nulls=False),
        _FLOAT32,
    )


class ThermalDemandFeatureGroup(FeatureGroup):
    """
    Computes heating demand and recent temperature from Helsinki temperature.

    ``heating_degree_hours`` is the instantaneous heating load relative to the
    17 °C base temperature used by Finnish degree-day statistics.
    ``temp_rolling_24h`` and ``temp_rolling_72h`` are trailing means of the
    Helsinki temperature, capturing the thermal inertia that keeps demand high
    after a cold snap. Both rolling windows look only backwards, so no feature
    leaks future observations. Null temperatures propagate to null heating
    degree hours; the rolling means skip nulls within their window and stay
    null until a full window of history exists.
    """

    name = THERMAL_DEMAND_GROUP
    specs = (
        FeatureSpec(
            name=HEATING_DEGREE_HOURS,
            inputs=(TEMPERATURE_COLUMN,),
            description=(
                "Hourly heating demand in degree-hours: degrees Celsius the "
                "Helsinki temperature falls below the 17 °C base, "
                "max(0, 17 - T); zero on mild hours."
            ),
        ),
        FeatureSpec(
            name=TEMP_ROLLING_24H,
            inputs=(TEMPERATURE_COLUMN,),
            description=(
                "Trailing 24-hour mean Helsinki temperature in °C; nulls "
                "skipped; null until 24 hours of history exist."
            ),
        ),
        FeatureSpec(
            name=TEMP_ROLLING_72H,
            inputs=(TEMPERATURE_COLUMN,),
            description=(
                "Trailing 72-hour mean Helsinki temperature in °C; nulls "
                "skipped; null until 72 hours of history exist."
            ),
        ),
    )

    def validate_inputs(self, table: pa.Table) -> None:
        super().validate_inputs(table)
        self._require_numeric(table, TEMPERATURE_COLUMN)

    def _compute(self, table: pa.Table) -> pa.Table:
        temperature = table[TEMPERATURE_COLUMN]
        features = (
            (HEATING_DEGREE_HOURS, heating_degree_hours(temperature)),
            (TEMP_ROLLING_24H, trailing_mean(temperature, 24)),
            (TEMP_ROLLING_72H, trailing_mean(temperature, 72)),
        )
        for name, values in features:
            table = table.append_column(name, pc.cast(values, _FLOAT32))
        return table

    def diagnostics(self, table: pa.Table) -> dict[str, Any]:
        """
        Summarises the temperature input and the heating demand it implies.

        Call it on the raw source table before building features.
        """
        self._require_columns(table, (TEMPERATURE_COLUMN,))
        self._require_numeric(table, TEMPERATURE_COLUMN)

        temperature = table[TEMPERATURE_COLUMN]
        degree_hours = heating_degree_hours(temperature)
        heating = pc.fill_null(pc.greater(degree_hours, _ZERO32), False)
        return {
            "temperature_nulls": temperature.null_count,
            "heating_hours": int(
                pc.sum(pc.cast(heating, pa.int64())).as_py() or 0
            ),
            "max_heating_degree_hours": (
                None
                if degree_hours.null_count == len(degree_hours)
                else pc.max(degree_hours).as_py()
            ),
        }