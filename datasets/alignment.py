"""Helpers for validating and merging fetcher outputs on one UTC hourly grid."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime, timedelta

import pyarrow as pa

from datasets.utils import utc_window

__all__ = [
    "align_hourly",
    "hourly_index",
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
