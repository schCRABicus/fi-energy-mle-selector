"""Base fetcher for Fingrid datasets that expose one numeric time series."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timedelta
from pathlib import Path
from typing import Self

import pyarrow as pa

from fi_energy.datasets.api import FetchRequest
from fi_energy.datasets.fingrid.client import FingridClient
from fi_energy.datasets.utils import iter_local_samples, mean_by_bucket, resolution_step, utc_window


class FingridSeriesFetcher:
    """
    Fetches one Fingrid time series and shapes it into a typed Arrow table.

    Subclasses supply the dataset id, the output value column, its schema, and
    an optional local Parquet snapshot.
    """

    RESOLUTIONS: dict[str, timedelta] = {
        "15min": timedelta(minutes=15),
        "30min": timedelta(minutes=30),
        "1h": timedelta(hours=1),
    }

    def __init__(
        self,
        api_key: str,
        *,
        dataset_id: int,
        dataset_name: str | None = None,
        value_column: str,
        schema: pa.Schema,
        local_path: str | Path | None,
        client: FingridClient | None = None,
        timeout: float = 30.0,
        max_retries: int = 5,
    ) -> None:
        self._dataset_id = dataset_id
        self._dataset_name = dataset_name
        self._value_column = value_column
        self._schema = schema
        self._client = client or FingridClient(
            api_key, timeout=timeout, max_retries=max_retries
        )
        self._local_path = Path(local_path) if local_path is not None else None

    @property
    def source_name(self) -> str:
        return f"fingrid_{self._dataset_id}"

    @property
    def dataset_name(self) -> str:
        return self._dataset_name or self.source_name

    @property
    def expected_schema(self) -> pa.Schema:
        return self._schema

    def fetch(self, request: FetchRequest) -> pa.Table:
        start, end = utc_window(request.start_utc, request.end_utc)
        return self._build_table(
            self._client.iter_samples(self._dataset_id, start, end),
            resolution=request.resolution,
            start=start,
            end=end,
        )

    def fetch_from_local(self, request: FetchRequest) -> pa.Table:
        start, end = utc_window(request.start_utc, request.end_utc)
        local_path = self._local_path
        if local_path is not None and local_path.exists():
            samples = iter_local_samples(local_path, value_column=self._value_column)
        else:
            samples = self._client.iter_samples(self._dataset_id, start, end)
        return self._build_table(
            samples,
            resolution=request.resolution,
            start=start,
            end=end,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def _build_table(
        self,
        samples: Iterable[tuple[datetime, float | None]],
        *,
        resolution: str,
        start: datetime,
        end: datetime,
    ) -> pa.Table:
        rows = mean_by_bucket(
            samples,
            start=start,
            end=end,
            step=resolution_step(self.RESOLUTIONS, resolution),
        )
        return pa.Table.from_pydict(
            {
                "timestamp_utc": [timestamp for timestamp, _ in rows],
                self._value_column: [value for _, value in rows],
            },
            schema=self._schema,
        )