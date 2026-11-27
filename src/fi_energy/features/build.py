"""Assembles the derived-feature table from the cleaned Phase 1 dataset."""

from __future__ import annotations

from dataclasses import asdict
from datetime import timedelta
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc

from fi_energy.datasets.alignment import TIMESTAMP_COLUMN, validate_hourly_table
from fi_energy.features.catalog import DEFAULT_CATALOG, FeatureCatalog

__all__ = ["build_feature_table", "describe_features"]

_TIMESTAMP_TYPE = pa.timestamp("us", tz="UTC")


def build_feature_table(
    source: pa.Table,
    catalog: FeatureCatalog = DEFAULT_CATALOG,
) -> pa.Table:
    """
    Builds the timestamped feature table from the cleaned Phase 1 table.

    Feature groups are applied in ``catalog`` order, so a group can consume
    features produced by any earlier group. The output contains
    ``timestamp_utc`` followed by every feature in the catalog; source columns
    are not duplicated. Row count and timestamp order match the input. The
    input table's schema metadata is preserved.

    Raises:
        TypeError: If ``source`` is not a pyarrow Table.
        ValueError: If the source is empty or its timestamp index is missing,
            null, duplicated, off the UTC hour, or not ``timestamp[us, UTC]``.
        RuntimeError: If a group violates the output contract.
    """
    _require_table(source, label="source")
    if TIMESTAMP_COLUMN not in source.column_names:
        raise ValueError(f"Source table has no {TIMESTAMP_COLUMN!r} column.")
    if source.num_rows == 0:
        raise ValueError("Source table has no rows.")

    timestamp_type = source.schema.field(TIMESTAMP_COLUMN).type
    if not timestamp_type.equals(_TIMESTAMP_TYPE):
        raise ValueError(
            f"{TIMESTAMP_COLUMN!r} must be {_TIMESTAMP_TYPE}, got "
            f"{timestamp_type}."
        )
    timestamps = source[TIMESTAMP_COLUMN].to_pylist()
    if any(timestamp is None for timestamp in timestamps):
        raise ValueError(f"{TIMESTAMP_COLUMN!r} contains null values.")
    validate_hourly_table(
        source,
        start=min(timestamps),
        end=max(timestamps) + timedelta(hours=1),
        label="source",
    )

    table = catalog.compute(source)
    features = table.select([TIMESTAMP_COLUMN, *catalog.feature_names])
    _validate_output(catalog, source, features)
    return features


def describe_features(
    source: pa.Table,
    features: pa.Table,
    catalog: FeatureCatalog = DEFAULT_CATALOG,
) -> dict[str, Any]:
    """
    Returns JSON-serialisable lineage, policy, and distribution diagnostics.

    ``source`` is the Phase 1 input; ``features`` is the output of
    ``build_feature_table`` built with the same ``catalog``. Per-feature
    statistics skip nulls. Group membership is composed into each feature
    entry; columns produced mid-pipeline are excluded from
    ``input_null_counts`` because they do not exist in the source.
    """
    _require_table(source, label="source")
    _require_table(features, label="features")
    input_columns = tuple(
        dict.fromkeys(
            name
            for spec in catalog.specs
            for name in spec.inputs
            if name in source.column_names
        )
    )
    return {
        "rows": features.num_rows,
        "groups": [group.name for group in catalog.groups],
        "feature_columns": list(catalog.feature_names),
        "features": [
            {**asdict(spec), "group": group.name}
            for group in catalog.groups
            for spec in group.specs
        ],
        "statistics": {
            name: _statistics(features[name])
            for name in catalog.feature_names
        },
        "null_values": {
            name: features[name].null_count
            for name in catalog.feature_names
        },
        "input_null_counts": {
            name: source[name].null_count for name in input_columns
        },
        "diagnostics": catalog.diagnostics(source),
    }


def _statistics(column: pa.ChunkedArray) -> dict[str, float | int | None]:
    return {
        "null_count": column.null_count,
        "min": pc.min(column).as_py(),
        "max": pc.max(column).as_py(),
        "mean": pc.mean(column).as_py(),
    }


def _validate_output(
    catalog: FeatureCatalog,
    source: pa.Table,
    features: pa.Table,
) -> None:
    expected_columns = [TIMESTAMP_COLUMN, *catalog.feature_names]
    if features.column_names != expected_columns:
        raise RuntimeError(
            f"Feature table columns {features.column_names} do not match "
            f"{expected_columns}."
        )
    if features.num_rows != source.num_rows:
        raise RuntimeError(
            f"Feature table has {features.num_rows} rows but the source has "
            f"{source.num_rows}."
        )
    for name in catalog.feature_names:
        field = features.schema.field(name)
        if not field.type.equals(pa.float32()) or not field.nullable:
            raise RuntimeError(
                f"Feature {name!r} must be a nullable float32, got "
                f"{field.type} (nullable={field.nullable})."
            )


def _require_table(value: object, *, label: str) -> None:
    if not isinstance(value, pa.Table):
        raise TypeError(
            f"Expected {label} to be a pyarrow Table, got "
            f"{type(value).__name__}."
        )