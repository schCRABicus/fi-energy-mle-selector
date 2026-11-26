from __future__ import annotations

from pathlib import Path

import pyarrow as pa

from datasets.fingrid.client import FingridClient
from datasets.fingrid.series_fetcher import FingridSeriesFetcher

DEFAULT_LOCAL_PATH = Path(__file__).parent / "data" / "nuclear_sample.parquet"

NUCLEAR_SCHEMA = pa.schema([
    pa.field("timestamp_utc", pa.timestamp("us", tz="UTC"), nullable=False),
    pa.field("nuclear_mw", pa.float32(), nullable=False),
])


class FingridNuclearFetcher(FingridSeriesFetcher):
    """
    Fetches Finnish nuclear power production (MW) from Fingrid dataset 188.

    Samples are averaged into the bucket size given by ``FetchRequest.resolution``.
    """

    def __init__(
        self,
        api_key: str,
        dataset_id: int = 188,
        *,
        client: FingridClient | None = None,
        local_path: str | Path | None = DEFAULT_LOCAL_PATH,
        timeout: float = 30.0,
        max_retries: int = 5,
    ) -> None:
        super().__init__(
            api_key,
            dataset_id=dataset_id,
            dataset_name="nuclear_production",
            value_column="nuclear_mw",
            schema=NUCLEAR_SCHEMA,
            local_path=local_path,
            client=client,
            timeout=timeout,
            max_retries=max_retries,
        )