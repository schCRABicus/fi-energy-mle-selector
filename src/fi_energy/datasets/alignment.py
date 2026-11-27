"""Helpers for validating, merging, and cleaning outputs on one UTC hourly grid."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import UTC, datetime, timedelta

import pyarrow as pa

from fi_energy.datasets.utils import utc_window

__all__ = [
    "align_hourly",
    "drop_incomplete_boundary_rows",
    "hourly_index",
    "interpolate_short_gaps",
    "trailing_window",
    "validate_hourly_table",
    "window_label",
]

TIMESTAMP_COLUMN = "timestamp_utc"
HOURLY_STEP = timedelta(hours=1)
_TIMESTAMP_TYPE = pa.timestamp("us", tz="UTC")


def trailing_window(
    now: datetime | None = None,
    *,
    years: int = 3,
) -> tuple[datetime, datetime]:
    """Returns a ``years``-long half-open window ending at the current UTC hour."""
    if years < 1:
        raise ValueError(f"years must be at least 1, got {years}.")

    current = datetime.now(UTC) if now is None else now
    if current.tzinfo is None or current.utcoffset() is None:
        raise ValueError("now must be timezone-aware.")
    end = current.astimezone(UTC).replace(minute=0, second=0, microsecond=0)
    try:
        start = end.replace(year=end.year - years)
    except ValueError:  # 29 February in a non-leap target year.
        start = end.replace(year=end.year - years, day=28)
    return start, end


def hourly_index(start: datetime, end: datetime) -> list[datetime]:
    """Returns every exact UTC hour in the half-open window ``[start, end)``."""
    start, end = _hourly_window(start, end)
    hours = int((end - start) / HOURLY_STEP)
    return [start + hour * HOURLY_STEP for hour in range(hours)]


def window_label(start: datetime, end: datetime) -> str:
    """
    Returns a short filename label such as ``3y`` for the window.

    Calendar-year windows use ``<n>y`` (leap-day aware); other windows use the
    exact duration in hours, e.g. ``24h``.
    """
    start, end = _hourly_window(start, end)
    for years in range(1, 101):
        if trailing_window(end, years=years)[0] == start:
            return f"{years}y"
    return f"{int((end - start) / HOURLY_STEP)}h"


def validate_hourly_table(
    table: pa.Table,
    *,
    start: datetime,
    end: datetime,
    label: str = "table",
) -> None:
    """Checks that a table sits on the UTC hourly grid inside ``[start, end)``."""
    _validated_timestamps(table, start=start, end=end, label=label)


def align_hourly(
    tables: Mapping[str, pa.Table],
    *,
    start: datetime,
    end: datetime,
) -> pa.Table:
    """
    Outer-joins tables onto every hour in ``[start, end)``.

    Each input must have a UTC ``timestamp_utc`` column and value columns with
    names that are unique across all tables. Missing hours and missing values
    become nulls; no interpolation is performed. Value columns are float32.
    """
    timestamps = hourly_index(start, end)
    position = {timestamp: index for index, timestamp in enumerate(timestamps)}

    columns: dict[str, list[float | None]] = {}
    for label, table in tables.items():
        for name in table.column_names:
            if name == TIMESTAMP_COLUMN:
                continue
            if name in columns:
                raise ValueError(
                    f"{label}: value column {name!r} appears in more than one table; "
                    "rename columns before aligning."
                )
            columns[name] = [None] * len(timestamps)

    for label, table in tables.items():
        normalized = _validated_timestamps(table, start=start, end=end, label=label)
        value_columns = [
            name for name in table.column_names if name != TIMESTAMP_COLUMN
        ]
        values = {name: table[name].to_pylist() for name in value_columns}
        for row, timestamp in enumerate(normalized):
            destination = position[timestamp]
            for name in value_columns:
                columns[name][destination] = values[name][row]

    schema = pa.schema([
        pa.field(TIMESTAMP_COLUMN, _TIMESTAMP_TYPE, nullable=False),
        *(pa.field(name, pa.float32()) for name in columns),
    ])
    return pa.Table.from_pydict(
        {TIMESTAMP_COLUMN: timestamps, **columns},
        schema=schema,
    )


def interpolate_short_gaps(
    table: pa.Table,
    *,
    max_gap: int = 2,
    columns: Iterable[str] | None = None,
) -> pa.Table:
    """
    Linearly interpolates short interior runs of missing hourly values.

    A run is filled only when it is at most ``max_gap`` rows long and has a
    non-null neighbour on both sides; with the default of 2, gaps shorter than
    three hours are filled. Leading/trailing gaps and longer interior gaps are
    left untouched (trim those with ``drop_incomplete_boundary_rows``). The
    timestamp column, column order, schema types, and metadata are preserved.
    """
    if max_gap < 0:
        raise ValueError(f"max_gap must be non-negative, got {max_gap}.")

    value_columns = _selected_value_columns(table, columns)
    updated = {
        name: pa.array(
            _interpolate_values(table[name].to_pylist(), max_gap),
            type=table.schema.field(name).type,
        )
        for name in value_columns
    }
    return pa.Table.from_arrays(
        [
            table[field.name] if field.name not in updated else updated[field.name]
            for field in table.schema
        ],
        schema=table.schema,
    )


def drop_incomplete_boundary_rows(
    table: pa.Table,
    *,
    columns: Iterable[str] | None = None,
) -> pa.Table:
    """
    Removes leading and trailing rows where any selected value column is null.

    Interior rows are never removed, even when a column is null there. This
    trims only the window edges where interpolation has no neighbour to use.
    """
    value_columns = _selected_value_columns(table, columns)
    if not value_columns:
        return table

    values = {name: table[name].to_pylist() for name in value_columns}
    first = 0
    last = table.num_rows
    while first < last and _row_has_null(values, value_columns, first):
        first += 1
    while last > first and _row_has_null(values, value_columns, last - 1):
        last -= 1
    if first == 0 and last == table.num_rows:
        return table
    return table.slice(first, last - first)


def _interpolate_values(
    values: list[float | None], max_gap: int
) -> list[float | None]:
    if max_gap == 0:
        return values

    result = list(values)
    index = 0
    while index < len(result):
        if result[index] is not None:
            index += 1
            continue

        gap_end = index
        while gap_end < len(result) and result[gap_end] is None:
            gap_end += 1
        gap = gap_end - index
        left = result[index - 1] if index > 0 else None
        right = result[gap_end] if gap_end < len(result) else None
        if gap <= max_gap and left is not None and right is not None:
            span = gap + 1
            for offset in range(1, gap + 1):
                fraction = offset / span
                result[index + offset - 1] = left + (right - left) * fraction
        index = gap_end
    return result


def _selected_value_columns(
    table: pa.Table, columns: Iterable[str] | None
) -> list[str]:
    if columns is None:
        return [
            name for name in table.column_names if name != TIMESTAMP_COLUMN
        ]
    selected = list(dict.fromkeys(columns))
    unknown = [
        name
        for name in selected
        if name == TIMESTAMP_COLUMN or name not in table.column_names
    ]
    if unknown:
        raise ValueError(
            f"Unknown value column(s) for cleaning: {', '.join(unknown)}."
        )
    return selected


def _row_has_null(
    values: Mapping[str, list[float | None]],
    columns: Iterable[str],
    row: int,
) -> bool:
    return any(values[name][row] is None for name in columns)


def _validated_timestamps(
    table: pa.Table,
    *,
    start: datetime,
    end: datetime,
    label: str,
) -> list[datetime]:
    start, end = _hourly_window(start, end)
    if TIMESTAMP_COLUMN not in table.column_names:
        raise ValueError(f"{label}: table has no {TIMESTAMP_COLUMN!r} column.")
    field = table.schema.field(TIMESTAMP_COLUMN)
    if not field.type.equals(_TIMESTAMP_TYPE):
        raise ValueError(
            f"{label}: {TIMESTAMP_COLUMN!r} must be {_TIMESTAMP_TYPE}, "
            f"got {field.type}."
        )

    normalized: list[datetime] = []
    seen: set[datetime] = set()
    for raw_timestamp in table[TIMESTAMP_COLUMN].to_pylist():
        if raw_timestamp is None:
            raise ValueError(f"{label}: {TIMESTAMP_COLUMN!r} contains a null value.")
        timestamp = raw_timestamp.astimezone(UTC)
        if timestamp.minute or timestamp.second or timestamp.microsecond:
            raise ValueError(
                f"{label}: timestamp {timestamp.isoformat()} is not on the hourly grid."
            )
        if not start <= timestamp < end:
            raise ValueError(
                f"{label}: timestamp {timestamp.isoformat()} lies outside "
                f"[{start.isoformat()}, {end.isoformat()})."
            )
        if timestamp in seen:
            raise ValueError(
                f"{label}: duplicate timestamp {timestamp.isoformat()}."
            )
        seen.add(timestamp)
        normalized.append(timestamp)
    return normalized


def _hourly_window(start: datetime, end: datetime) -> tuple[datetime, datetime]:
    start, end = utc_window(start, end)
    start = _require_hour(start, "start_utc")
    end = _require_hour(end, "end_utc")
    return start, end


def _require_hour(value: datetime, field: str) -> datetime:
    if value.minute or value.second or value.microsecond:
        raise ValueError(
            f"{field} must be on an exact UTC hour, got {value.isoformat()}."
        )
    return value
