from __future__ import annotations

from pathlib import Path

import pyarrow as pa

from datasets.fingrid.client import FingridClient
from datasets.fingrid.series_fetcher import FingridSeriesFetcher

DEFAULT_LOCAL_PATH = Path(__file__).parent / "data" / "consumption_sample.parquet"

CONSUMPTION_SCHEMA = pa.schema([
    pa.field("timestamp_utc", pa.timestamp("us", tz="UTC"), nullable=False),
    pa.field("consumption_mw", pa.float32(), nullable=False),
])


class FingridConsumptionFetcher(FingridSeriesFetcher):
    """
    Fetches Finnish electricity consumption from Fingrid dataset 124.

    Fingrid reports consumption in MWh/h, which is numerically equivalent to the
    average power in MW over the interval. Samples are averaged into the bucket
    size given by ``FetchRequest.resolution``.
    """

    def __init__(
        self,
        api_key: str,
        dataset_id: int = 124,
        *,
        client: FingridClient | None = None,
        local_path: str | Path | None = DEFAULT_LOCAL_PATH,
        timeout: float = 30.0,
        max_retries: int = 5,
    ) -> None:
        super().__init__(
            api_key,
            dataset_id=dataset_id,
            dataset_name="electricity_consumption",
            value_column="consumption_mw",
            schema=CONSUMPTION_SCHEMA,
            local_path=local_path,
            client=client,
            timeout=timeout,
            max_retries=max_retries,
        )