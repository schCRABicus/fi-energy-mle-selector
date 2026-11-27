"""Grid-ratio features derived from Finnish wind, consumption, and capacity."""

from __future__ import annotations

from typing import Any

import pyarrow as pa
import pyarrow.compute as pc

from fi_energy.features.ratios import safe_ratio
from fi_energy.features.spec import FeatureSpec

__all__ = [
    "GRID_RATIO_FEATURES",
    "GRID_RATIO_GROUP",
    "add_grid_ratio_features",
    "grid_ratio_diagnostics",
]

GRID_RATIO_GROUP = "grid_ratios"

WIND_COLUMN = "wind_mw"
CONSUMPTION_COLUMN = "consumption_mw"
WIND_CAPACITY_COLUMN = "wind_capacity_mw"
WIND_PENETRATION_RATIO = "wind_penetration_ratio"
WIND_CAPACITY_FACTOR = "wind_capacity_factor"

_FLOAT32 = pa.float32()
_ZERO = pa.scalar(0.0, type=_FLOAT32)

GRID_RATIO_FEATURES: tuple[FeatureSpec, ...] = (
    FeatureSpec(
        name=WIND_PENETRATION_RATIO,
        inputs=(WIND_COLUMN, CONSUMPTION_COLUMN),
        group=GRID_RATIO_GROUP,
        description=(
            "Instantaneous share of Finnish electricity consumption met by "
            "wind generation; can exceed 1 when surplus wind is exported."
        ),
    ),
    FeatureSpec(
        name=WIND_CAPACITY_FACTOR,
        inputs=(WIND_COLUMN, WIND_CAPACITY_COLUMN),
        group=GRID_RATIO_GROUP,
        description=(
            "Wind generation as a fraction of installed wind capacity, a "
            "physical utilisation rate between 0 and 1."
        ),
    ),
)


def add_grid_ratio_features(table: pa.Table) -> pa.Table:
    """
    Appends ``wind_penetration_ratio`` and ``wind_capacity_factor``.

    Negative wind generation is a measurement artefact (generation cannot be
    negative), so it is clipped to zero for both ratios; the source ``wind_mw``
    column is left untouched. Null inputs propagate to null features, and a
    zero denominator becomes null rather than infinity. Existing columns, their
    order, the timestamp field, and schema metadata are preserved.

    Raises:
        TypeError: If ``table`` is not a pyarrow Table.
        ValueError: If a required input is missing or non-numeric, or if an
            output feature column already exists.
    """
    _require_table(table)
    required_inputs = _input_columns()
    _require_columns(table, required_inputs)
    for name in required_inputs:
        _require_numeric(table, name)
    _require_absent_columns(table, _feature_columns())

    wind = _clipped_wind(table)
    features = (
        (WIND_PENETRATION_RATIO, safe_ratio(wind, table[CONSUMPTION_COLUMN])),
        (WIND_CAPACITY_FACTOR, safe_ratio(wind, table[WIND_CAPACITY_COLUMN])),
    )
    for name, values in features:
        table = table.append_column(name, values)
    return table


def grid_ratio_diagnostics(table: pa.Table) -> dict[str, Any]:
    """
    Summarises the data corrections applied by ``add_grid_ratio_features``.

    Returns the count and minimum of negative wind readings that would be
    clipped to zero, plus the number of zero denominators per ratio. Call it on
    the raw source table before building features.
    """
    _require_table(table)
    _require_columns(
        table,
        (WIND_COLUMN, CONSUMPTION_COLUMN, WIND_CAPACITY_COLUMN),
    )
    for name in (WIND_COLUMN, CONSUMPTION_COLUMN, WIND_CAPACITY_COLUMN):
        _require_numeric(table, name)

    wind = pc.cast(table[WIND_COLUMN], _FLOAT32)
    negative = pc.fill_null(pc.less(wind, _ZERO), False)
    negative_count = _count_true(negative)
    negative_values = pc.filter(wind, negative)
    negative_min = (
        pc.min(negative_values).as_py() if negative_count else None
    )

    zero_denominators = {
        WIND_PENETRATION_RATIO: _zero_count(table[CONSUMPTION_COLUMN]),
        WIND_CAPACITY_FACTOR: _zero_count(table[WIND_CAPACITY_COLUMN]),
    }
    return {
        "negative_wind_count": negative_count,
        "negative_wind_min_mw": negative_min,
        "zero_denominators": zero_denominators,
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


def _input_columns() -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            name for spec in GRID_RATIO_FEATURES for name in spec.inputs
        )
    )


def _feature_columns() -> tuple[str, ...]:
    return tuple(spec.name for spec in GRID_RATIO_FEATURES)


def _require_table(table: object) -> None:
    if not isinstance(table, pa.Table):
        raise TypeError(
            f"Expected a pyarrow Table, got {type(table).__name__}."
        )


def _require_columns(table: pa.Table, names: tuple[str, ...]) -> None:
    missing = [name for name in names if name not in table.column_names]
    if missing:
        raise ValueError(
            f"Table is missing required column(s): {', '.join(missing)}."
        )


def _require_absent_columns(table: pa.Table, names: tuple[str, ...]) -> None:
    present = [name for name in names if name in table.column_names]
    if present:
        raise ValueError(
            f"Table already has feature column(s): {', '.join(present)}."
        )


def _require_numeric(table: pa.Table, name: str) -> None:
    dtype = table.schema.field(name).type
    if not (pa.types.is_integer(dtype) or pa.types.is_floating(dtype)):
        raise ValueError(
            f"Column {name!r} must be numeric (integer or floating), got "
            f"{dtype}."
        )