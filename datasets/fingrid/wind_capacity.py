from __future__ import annotations

from pathlib import Path

import pyarrow as pa

from datasets.fingrid.client import FingridClient
from datasets.fingrid.series_fetcher import FingridSeriesFetcher

DEFAULT_LOCAL_PATH = Path(__file__).parent / "data" / "wind_capacity_sample.parquet"

CAPACITY_SCHEMA = pa.schema([
    pa.field("timestamp_utc", pa.timestamp("us", tz="UTC"), nullable=False),
    pa.field("wind_capacity_mw", pa.float32(), nullable=False),
])


class FingridWindCapacityFetcher(FingridSeriesFetcher):
    """
    Fetches Finnish installed wind power capacity (MW) from Fingrid dataset 268.

    Samples are averaged into the bucket size given by ``FetchRequest.resolution``.
    """

    def __init__(
        self,
        api_key: str,
        dataset_id: int = 268,
        *,
        client: FingridClient | None = None,
        local_path: str | Path | None = DEFAULT_LOCAL_PATH,
        timeout: float = 30.0,
        max_retries: int = 5,
    ) -> None:
        super().__init__(
            api_key,
            dataset_id=dataset_id,
            dataset_name="wind_capacity",
            value_column="wind_capacity_mw",
            schema=CAPACITY_SCHEMA,
            local_path=local_path,
            client=client,
            timeout=timeout,
            max_retries=max_retries,
        )