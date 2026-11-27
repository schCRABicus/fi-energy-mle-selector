"""Wind-ratio features derived from Finnish wind, consumption, and capacity."""

from __future__ import annotations

from typing import Any

import pyarrow as pa
import pyarrow.compute as pc

from fi_energy.features.group import FeatureGroup
from fi_energy.features.ratios import safe_ratio
from fi_energy.features.spec import FeatureSpec

__all__ = ["WindRatiosFeatureGroup"]

WIND_RATIOS_GROUP = "wind_ratios"

WIND_COLUMN = "wind_mw"
CONSUMPTION_COLUMN = "consumption_mw"
WIND_CAPACITY_COLUMN = "wind_capacity_mw"
WIND_PENETRATION_RATIO = "wind_penetration_ratio"
WIND_CAPACITY_FACTOR = "wind_capacity_factor"

_FLOAT32 = pa.float32()
_ZERO = pa.scalar(0.0, type=_FLOAT32)


class WindRatiosFeatureGroup(FeatureGroup):
    """
    Computes ``wind_penetration_ratio`` and ``wind_capacity_factor``.

    Negative wind generation is a measurement artefact (generation cannot be
    negative), so it is clipped to zero for both ratios; the source ``wind_mw``
    column is left untouched. Null inputs propagate to null features, and a
    zero denominator becomes null rather than infinity.
    """

    name = WIND_RATIOS_GROUP
    specs = (
        FeatureSpec(
            name=WIND_PENETRATION_RATIO,
            inputs=(WIND_COLUMN, CONSUMPTION_COLUMN),
            description=(
                "Instantaneous share of Finnish electricity consumption met "
                "by wind generation; can exceed 1 when surplus wind is "
                "exported."
            ),
        ),
        FeatureSpec(
            name=WIND_CAPACITY_FACTOR,
            inputs=(WIND_COLUMN, WIND_CAPACITY_COLUMN),
            description=(
                "Wind generation as a fraction of installed wind capacity, a "
                "physical utilisation rate between 0 and 1."
            ),
        ),
    )

    def validate_inputs(self, table: pa.Table) -> None:
        super().validate_inputs(table)
        for name in self.inputs:
            self._require_numeric(table, name)

    def _compute(self, table: pa.Table) -> pa.Table:
        wind = _clipped_wind(table)
        features = (
            (WIND_PENETRATION_RATIO, safe_ratio(wind, table[CONSUMPTION_COLUMN])),
            (WIND_CAPACITY_FACTOR, safe_ratio(wind, table[WIND_CAPACITY_COLUMN])),
        )
        for name, values in features:
            table = table.append_column(name, values)
        return table

    def diagnostics(self, table: pa.Table) -> dict[str, Any]:
        """
        Summarises the corrections applied while computing the ratios.

        Returns the count and minimum of negative wind readings that are
        clipped to zero, plus the number of zero denominators per ratio. Call
        it on the raw source table before building features.
        """
        self._require_columns(table, self.inputs)
        for name in self.inputs:
            self._require_numeric(table, name)

        wind = pc.cast(table[WIND_COLUMN], _FLOAT32)
        negative = pc.fill_null(pc.less(wind, _ZERO), False)
        negative_count = _count_true(negative)
        negative_values = pc.filter(wind, negative)
        negative_min = (
            pc.min(negative_values).as_py() if negative_count else None
        )

        return {
            "negative_wind_count": negative_count,
            "negative_wind_min_mw": negative_min,
            "zero_denominators": {
                WIND_PENETRATION_RATIO: _zero_count(table[CONSUMPTION_COLUMN]),
                WIND_CAPACITY_FACTOR: _zero_count(table[WIND_CAPACITY_COLUMN]),
            },
        }


def _clipped_wind(table: pa.Table) -> pa.ChunkedArray:
    wind = pc.cast(table[WIND_COLUMN], _FLOAT32)
    # A null comparison yields a null mask, and if_else then keeps the null,
    # so missing wind values are not fabricated as zero generation.
    return pc.cast(
        pc.if_else(pc.less(wind, _ZERO), _ZERO, wind),
        _FLOAT32,
    )


def _zero_count(column: pa.ChunkedArray) -> int:
    denominator = pc.cast(column, _FLOAT32)
    return _count_true(pc.fill_null(pc.equal(denominator, _ZERO), False))


def _count_true(mask: pa.ChunkedArray) -> int:
    return int(pc.sum(pc.cast(mask, pa.int64())).as_py() or 0)