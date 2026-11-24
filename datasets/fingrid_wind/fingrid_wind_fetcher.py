from __future__ import annotations

from collections import defaultdict
from datetime import UTC, datetime, timedelta
from typing import Self

import pyarrow as pa

from datasets.api import DataSourceFetcher, FetchRequest
from datasets.fingrid_wind.fingrid_client import FingridClient


class FingridWindFetcher:
    """
    Fetches Finnish wind power generation (MW) from Fingrid Open Data.

    Samples are averaged into the bucket size given by ``FetchRequest.resolution``.
    """

    RESOLUTIONS: dict[str, timedelta] = {
        "15min": timedelta(minutes=15),
        "30min": timedelta(minutes=30),
        "1h": timedelta(hours=1),
    }

    def __init__(
        self,
        api_key: str,
        dataset_id: int = 75,
        *,
        client: FingridClient | None = None,
        timeout: float = 30.0,
        max_retries: int = 5,
    ) -> None:
        self._dataset_id = dataset_id
        self._client = client or FingridClient(
            api_key, timeout=timeout, max_retries=max_retries
        )

    @property
    def source_name(self) -> str:
        return f"fingrid_{self._dataset_id}"

    @property
    def expected_schema(self) -> pa.Schema:
        return pa.schema([
            pa.field("timestamp_utc", pa.timestamp("us", tz="UTC"), nullable=False),
            pa.field("wind_mw", pa.float32(), nullable=False),
        ])

    def fetch(self, request: FetchRequest) -> pa.Table:
        start = self._require_utc(request.start_utc, "start_utc")
        end = self._require_utc(request.end_utc, "end_utc")
        if start >= end:
            raise ValueError(
                f"[{self.source_name}] start_utc ({start.isoformat()}) must be "
                f"before end_utc ({end.isoformat()})."
            )
        step = self._resolution_step(request.resolution)

        samples: defaultdict[datetime, list[float]] = defaultdict(list)
        for timestamp, value in self._client.iter_samples(self._dataset_id, start, end):
            if value is not None and start <= timestamp < end:
                samples[self._floor(timestamp, step)].append(value)

        rows = sorted(samples.items())
        return pa.Table.from_pydict(
            {
                "timestamp_utc": [timestamp for timestamp, _ in rows],
                "wind_mw": [sum(values) / len(values) for _, values in rows],
            },
            schema=self.expected_schema,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    @staticmethod
    def _require_utc(value: datetime, field: str) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError(f"FetchRequest.{field} must be timezone-aware (UTC).")
        return value.astimezone(UTC)

    @classmethod
    def _resolution_step(cls, resolution: str) -> timedelta:
        try:
            return cls.RESOLUTIONS[resolution]
        except KeyError:
            supported = ", ".join(sorted(cls.RESOLUTIONS))
            raise ValueError(
                f"Unsupported resolution {resolution!r}; expected one of: {supported}."
            ) from None

    @staticmethod
    def _floor(timestamp: datetime, step: timedelta) -> datetime:
        midnight = timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
        return midnight + (timestamp - midnight) // step * step


def run_fetcher(fetcher: DataSourceFetcher, request: FetchRequest) -> pa.Table:
    table = fetcher.fetch(request)
    if not table.schema.equals(fetcher.expected_schema, check_metadata=False):
        raise TypeError(
            f"[{fetcher.source_name}] Schema mismatch.\n"
            f"Expected: {fetcher.expected_schema}\nGot: {table.schema}"
        )
    return table