from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Self

import pyarrow as pa
import pyarrow.parquet as pq

from datasets.api import DataSourceFetcher, FetchRequest
from datasets.fingrid_wind.fingrid_client import FingridClient

DEFAULT_LOCAL_PATH = Path(__file__).parent / "data" / "fingrid_wind_sample.parquet"


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
        local_path: str | Path | None = DEFAULT_LOCAL_PATH,
        timeout: float = 30.0,
        max_retries: int = 5,
    ) -> None:
        self._dataset_id = dataset_id
        self._client = client or FingridClient(
            api_key, timeout=timeout, max_retries=max_retries
        )
        self._local_path = Path(local_path) if local_path is not None else None

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
        start, end = self._window(request)
        return self._build_table(
            self._client.iter_samples(self._dataset_id, start, end),
            request,
            start,
            end,
        )

    def fetch_from_local(self, request: FetchRequest) -> pa.Table:
        start, end = self._window(request)
        if self._local_path is not None and self._local_path.exists():
            samples = self._read_local()
        else:
            samples = self._client.iter_samples(self._dataset_id, start, end)
        return self._build_table(samples, request, start, end)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def _read_local(self) -> Iterator[tuple[datetime, float | None]]:
        table = pq.read_table(self._local_path, columns=["timestamp_utc", "wind_mw"])
        for timestamp, value in zip(
            table["timestamp_utc"].to_pylist(),
            table["wind_mw"].to_pylist(),
            strict=True,
        ):
            timestamp = timestamp.replace(tzinfo=UTC) if timestamp.tzinfo is None else timestamp
            yield timestamp.astimezone(UTC), None if value is None else float(value)

    def _build_table(
        self,
        samples: Iterable[tuple[datetime, float | None]],
        request: FetchRequest,
        start: datetime,
        end: datetime,
    ) -> pa.Table:
        step = self._resolution_step(request.resolution)

        buckets: defaultdict[datetime, list[float]] = defaultdict(list)
        for timestamp, value in samples:
            if value is not None and start <= timestamp < end:
                buckets[self._floor(timestamp, step)].append(value)

        rows = sorted(buckets.items())
        return pa.Table.from_pydict(
            {
                "timestamp_utc": [timestamp for timestamp, _ in rows],
                "wind_mw": [sum(values) / len(values) for _, values in rows],
            },
            schema=self.expected_schema,
        )

    @staticmethod
    def _window(request: FetchRequest) -> tuple[datetime, datetime]:
        start = FingridWindFetcher._require_utc(request.start_utc, "start_utc")
        end = FingridWindFetcher._require_utc(request.end_utc, "end_utc")
        if start >= end:
            raise ValueError(
                f"start_utc ({start.isoformat()}) must be before end_utc ({end.isoformat()})."
            )
        return start, end

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