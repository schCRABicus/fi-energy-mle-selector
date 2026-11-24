from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, runtime_checkable

import pyarrow as pa

__all__ = ["DataSourceFetcher", "FetchRequest"]


@dataclass(frozen=True)
class FetchRequest:
    """Immutable query parameters passed to any datasource fetcher."""
    start_utc: datetime
    end_utc: datetime
    resolution: str = "1h"  # e.g., "15min", "1h"


@runtime_checkable
class DataSourceFetcher(Protocol):
    """Contract that all external/local data adapters must satisfy."""

    @property
    def source_name(self) -> str:
        """Unique identifier for logging and lineage (e.g., 'fingrid_75')."""
        ...

    @property
    def expected_schema(self) -> pa.Schema:
        """
        Strict PyArrow schema enforced on output.
        Must always include a timezone-aware 'timestamp_utc' (pa.timestamp('us', tz='UTC'))
        and typed feature columns (e.g., pa.float32()).
        """
        ...

    def fetch(self, request: FetchRequest) -> pa.Table:
        """
        Fetches raw data for [start_utc, end_utc), normalizes timestamps to UTC,
        casts columns to `expected_schema`, and returns a validated PyArrow Table.
        """
        ...

    def fetch_from_local(self, request: FetchRequest) -> pa.Table:
        """
        Returns data from a bundled/local snapshot when one is available, otherwise
        falls back to `fetch`. Output follows the same `expected_schema` contract.
        """
        ...