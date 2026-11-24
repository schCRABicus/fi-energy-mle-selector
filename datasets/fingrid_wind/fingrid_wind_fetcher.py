from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timedelta
from pathlib import Path
from typing import Self

import pyarrow as pa

from datasets.api import FetchRequest
from datasets.fingrid_wind.fingrid_client import FingridClient
from datasets.utils import iter_local_samples, mean_by_bucket, resolution_step, utc_window

DEFAULT_LOCAL_PATH = Path(__file__).parent / "data" / "fingrid_wind_sample.parquet"

WIND_SCHEMA = pa.schema([
    pa.field("timestamp_utc", pa.timestamp("us", tz="UTC"), nullable=False),
    pa.field("wind_mw", pa.float32(), nullable=False),
])


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
        return WIND_SCHEMA

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
            samples = iter_local_samples(local_path, value_column="wind_mw")
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
                "wind_mw": [value for _, value in rows],
            },
            schema=self.expected_schema,
        )