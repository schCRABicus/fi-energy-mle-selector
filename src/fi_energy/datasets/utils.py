"""Source-agnostic helpers shared by dataset clients and fetchers."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pyarrow.parquet as pq

__all__ = [
    "iter_local_samples",
    "mean_by_bucket",
    "resolution_step",
    "utc_window",
]


def utc_window(start: datetime, end: datetime) -> tuple[datetime, datetime]:
    """Validates a half-open window and normalizes both bounds to UTC."""
    start = _require_utc(start, "start_utc")
    end = _require_utc(end, "end_utc")
    if start >= end:
        raise ValueError(
            f"start_utc ({start.isoformat()}) must be before end_utc ({end.isoformat()})."
        )
    return start, end


def resolution_step(resolutions: Mapping[str, timedelta], resolution: str) -> timedelta:
    """Looks up a named resolution, raising a helpful error for unknown names."""
    try:
        return resolutions[resolution]
    except KeyError:
        supported = ", ".join(sorted(resolutions))
        raise ValueError(
            f"Unsupported resolution {resolution!r}; expected one of: {supported}."
        ) from None


def mean_by_bucket(
    samples: Iterable[tuple[datetime, float | None]],
    *,
    start: datetime,
    end: datetime,
    step: timedelta,
) -> list[tuple[datetime, float]]:
    """Averages non-null samples into buckets of ``step`` inside ``[start, end)``."""
    buckets: defaultdict[datetime, list[float]] = defaultdict(list)
    for timestamp, value in samples:
        if value is not None and start <= timestamp < end:
            buckets[floor_timestamp(timestamp, step)].append(value)
    return [
        (timestamp, sum(values) / len(values))
        for timestamp, values in sorted(buckets.items())
    ]


def iter_local_samples(
    path: str | Path,
    *,
    value_column: str,
    timestamp_column: str = "timestamp_utc",
) -> Iterator[tuple[datetime, float | None]]:
    """Reads ``(timestamp_utc, value)`` samples from a local Parquet snapshot."""
    table = pq.read_table(path, columns=[timestamp_column, value_column])
    for timestamp, value in zip(
        table[timestamp_column].to_pylist(),
        table[value_column].to_pylist(),
        strict=True,
    ):
        timestamp = timestamp.replace(tzinfo=UTC) if timestamp.tzinfo is None else timestamp
        yield timestamp.astimezone(UTC), None if value is None else float(value)


def _require_utc(value: datetime, field: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field} must be timezone-aware (UTC).")
    return value.astimezone(UTC)


def floor_timestamp(timestamp: datetime, step: timedelta) -> datetime:
    """Floors a timestamp to the start of its ``step`` bucket on the same day."""
    midnight = timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
    return midnight + (timestamp - midnight) // step * step