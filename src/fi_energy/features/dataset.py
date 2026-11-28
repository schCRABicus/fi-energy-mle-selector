"""Assemble the 24-feature training dataset and its next-hour target."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc

from fi_energy.datasets.alignment import TIMESTAMP_COLUMN
from fi_energy.features.build import build_feature_table
from fi_energy.features.catalog import DEFAULT_CATALOG, FeatureCatalog

__all__ = [
    "FEATURE_COLUMNS",
    "RAW_FEATURE_COLUMNS",
    "SOURCE_TARGET_COLUMN",
    "TARGET_COLUMN",
    "DatasetBuild",
    "build_training_dataset",
]

SOURCE_TARGET_COLUMN = "spot_price_c_kwh"
TARGET_COLUMN = "spot_price_next_hour_c_kwh"

# Raw phase-1 fundamentals. Temperatures cover the three weather regions;
# Helsinki radiation stands in for solar generation. Sparse wind-speed columns
# are deliberately excluded so a complete-case dataset keeps every recent hour.
RAW_FEATURE_COLUMNS: tuple[str, ...] = (
    "wind_mw",
    "wind_capacity_mw",
    "consumption_mw",
    "nuclear_mw",
    "hydro_mw",
    "se1_flow_mw",
    "se3_flow_mw",
    "ee_flow_mw",
    "helsinki_temperature_c",
    "vaasa_temperature_c",
    "oulu_temperature_c",
    "helsinki_radiation_wm2",
)

DERIVED_FEATURE_COLUMNS: tuple[str, ...] = DEFAULT_CATALOG.feature_names
FEATURE_COLUMNS: tuple[str, ...] = (
    RAW_FEATURE_COLUMNS + DERIVED_FEATURE_COLUMNS
)

if len(FEATURE_COLUMNS) != 24:
    raise RuntimeError(
        f"The training dataset must have 24 features, got "
        f"{len(FEATURE_COLUMNS)}."
    )

_FLOAT32 = pa.float32()


@dataclass(frozen=True, slots=True)
class DatasetBuild:
    """A complete-case training table plus the leakage checks that passed."""

    table: pa.Table
    source_rows: int
    dropped_rows: int
    feature_columns: tuple[str, ...]
    target_column: str
    checks: dict[str, bool]


def build_training_dataset(
    source: pa.Table,
    catalog: FeatureCatalog = DEFAULT_CATALOG,
) -> DatasetBuild:
    """
    Builds ``timestamp_utc`` + 24 features + the next-hour spot-price target.

    The target for row ``t`` is the spot price at ``t + 1 hour``; every feature
    is observed at ``t``. Rows with a null or NaN feature/target are dropped,
    so the result contains no missing values and no infinite values. The
    returned checks confirm that the target is absent from the features, that
    every feature is unchanged when future source rows are removed, and that
    the target is exactly the one-hour-ahead spot price.

    Raises:
        TypeError: If ``source`` is not a pyarrow Table.
        ValueError: If a required source column is missing or a feature
            specification would consume the target.
        RuntimeError: If the assembled table violates the dataset contract.
    """
    _require_source_columns(source)
    features = build_feature_table(source, catalog)
    feature_columns = RAW_FEATURE_COLUMNS + tuple(catalog.feature_names)

    missing = [
        name for name in RAW_FEATURE_COLUMNS if name not in source.column_names
    ]
    if missing:
        raise ValueError(
            "Source is missing raw feature column(s): " + ", ".join(missing)
        )
    _require_target_is_not_a_feature(catalog)

    target = _next_hour_target(source[SOURCE_TARGET_COLUMN])
    arrays = [
        source[TIMESTAMP_COLUMN],
        *(source[name] for name in RAW_FEATURE_COLUMNS),
        *(features[name] for name in catalog.feature_names),
        target,
    ]
    names = [
        TIMESTAMP_COLUMN,
        *RAW_FEATURE_COLUMNS,
        *catalog.feature_names,
        TARGET_COLUMN,
    ]
    assembled = pa.Table.from_arrays(arrays, names=names)

    mask = _complete_case_mask(assembled, feature_columns)
    table = assembled.filter(mask)
    if table.num_rows == 0:
        raise RuntimeError("Every dataset row was dropped as incomplete.")
    _validate_complete(table, feature_columns)

    checks = {
        "target_absent_from_features": TARGET_COLUMN not in feature_columns
        and SOURCE_TARGET_COLUMN not in feature_columns,
        "target_absent_from_specs": _target_absent_from_specs(catalog),
        "features_ignore_future_rows": _prefix_matches_full(
            source, catalog
        ),
        "target_is_next_hour": _target_is_next_hour(
            target, source[SOURCE_TARGET_COLUMN]
        ),
    }
    if not all(checks.values()):
        failed = ", ".join(name for name, ok in checks.items() if not ok)
        raise RuntimeError(f"Dataset leakage check(s) failed: {failed}.")

    return DatasetBuild(
        table=table,
        source_rows=source.num_rows,
        dropped_rows=source.num_rows - table.num_rows,
        feature_columns=feature_columns,
        target_column=TARGET_COLUMN,
        checks=checks,
    )


def feature_statistics(build: DatasetBuild) -> dict[str, dict[str, Any]]:
    """Returns null, min, mean, and max for every feature and the target."""
    names = (*build.feature_columns, build.target_column)
    return {
        name: {
            "null_count": build.table[name].null_count,
            "min": pc.min(build.table[name]).as_py(),
            "mean": pc.mean(build.table[name]).as_py(),
            "max": pc.max(build.table[name]).as_py(),
        }
        for name in names
    }


def _require_source_columns(source: pa.Table) -> None:
    if not isinstance(source, pa.Table):
        raise TypeError(
            f"Expected source to be a pyarrow Table, got "
            f"{type(source).__name__}."
        )
    required = {TIMESTAMP_COLUMN, SOURCE_TARGET_COLUMN, *RAW_FEATURE_COLUMNS}
    missing = sorted(required - set(source.column_names))
    if missing:
        raise ValueError(
            "Source is missing required column(s): " + ", ".join(missing)
        )
    target_type = source.schema.field(SOURCE_TARGET_COLUMN).type
    if not pa.types.is_floating(target_type):
        raise ValueError(
            f"{SOURCE_TARGET_COLUMN!r} must be floating, got {target_type}."
        )


def _require_target_is_not_a_feature(catalog: FeatureCatalog) -> None:
    if not _target_absent_from_specs(catalog):
        raise ValueError(
            f"A feature specification consumes {SOURCE_TARGET_COLUMN!r} or "
            f"{TARGET_COLUMN!r}; the target cannot be a model input."
        )


def _target_absent_from_specs(catalog: FeatureCatalog) -> bool:
    forbidden = {SOURCE_TARGET_COLUMN, TARGET_COLUMN}
    return all(
        forbidden.isdisjoint(spec.inputs) for spec in catalog.specs
    )


def _next_hour_target(spot: pa.ChunkedArray) -> pa.Array:
    values = spot.to_pylist()
    return pa.array(values[1:] + [None], type=_FLOAT32)


def _complete_case_mask(
    table: pa.Table,
    feature_columns: tuple[str, ...],
) -> pa.ChunkedArray:
    names = [*feature_columns, TARGET_COLUMN]
    mask: pa.ChunkedArray | None = None
    for name in names:
        column = table[name]
        valid = pc.is_valid(column)
        if pa.types.is_floating(column.type):
            valid = pc.and_(
                valid,
                pc.fill_null(pc.invert(pc.is_nan(column)), False),
            )
        mask = valid if mask is None else pc.and_(mask, valid)
    assert mask is not None
    return mask


def _validate_complete(
    table: pa.Table,
    feature_columns: tuple[str, ...],
) -> None:
    expected = [TIMESTAMP_COLUMN, *feature_columns, TARGET_COLUMN]
    if table.column_names != expected:
        raise RuntimeError(
            f"Dataset columns {table.column_names} do not match {expected}."
        )
    for name in expected[1:]:
        if table[name].null_count:
            raise RuntimeError(
                f"Dataset column {name!r} still has "
                f"{table[name].null_count:,} null values."
            )
        if pa.types.is_floating(table[name].type):
            nan_count = int(
                pc.sum(
                    pc.cast(
                        pc.fill_null(pc.is_nan(table[name]), False), pa.int64()
                    )
                ).as_py()
                or 0
            )
            if nan_count:
                raise RuntimeError(
                    f"Dataset column {name!r} still has "
                    f"{nan_count:,} NaN values."
                )


def _prefix_matches_full(
    source: pa.Table,
    catalog: FeatureCatalog,
) -> bool:
    """Recomputing on a prefix must not change any earlier feature value."""
    cut = max(100, source.num_rows // 2)
    if cut >= source.num_rows:
        return True
    full = build_feature_table(source, catalog)
    prefix = build_feature_table(source.slice(0, cut), catalog)
    return all(
        full[name].slice(0, cut).to_pylist() == prefix[name].to_pylist()
        for name in DERIVED_FEATURE_COLUMNS
    )


def _target_is_next_hour(
    target: pa.Array,
    spot: pa.ChunkedArray,
) -> bool:
    target_values = target.to_pylist()
    spot_values = spot.to_pylist()
    return (
        target_values[-1] is None
        and target_values[:-1] == spot_values[1:]
    )