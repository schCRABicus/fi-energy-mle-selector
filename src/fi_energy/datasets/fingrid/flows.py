"""Cross-border electricity flows between Finland and its neighbours.

Sign convention follows Fingrid: positive values mean transmission from Finland
to the neighbouring bidding zone, negative values mean transmission into Finland.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa

from fi_energy.datasets.fingrid.client import FingridClient
from fi_energy.datasets.fingrid.series_fetcher import FingridSeriesFetcher

DATA_DIR = Path(__file__).parent / "data"

DEFAULT_SE1_LOCAL_PATH = DATA_DIR / "se1_flow_sample.parquet"
DEFAULT_SE3_LOCAL_PATH = DATA_DIR / "se3_flow_sample.parquet"
DEFAULT_EE_LOCAL_PATH = DATA_DIR / "ee_flow_sample.parquet"

SE1_FLOW_SCHEMA = pa.schema([
    pa.field("timestamp_utc", pa.timestamp("us", tz="UTC"), nullable=False),
    pa.field("se1_flow_mw", pa.float32(), nullable=False),
])

SE3_FLOW_SCHEMA = pa.schema([
    pa.field("timestamp_utc", pa.timestamp("us", tz="UTC"), nullable=False),
    pa.field("se3_flow_mw", pa.float32(), nullable=False),
])

EE_FLOW_SCHEMA = pa.schema([
    pa.field("timestamp_utc", pa.timestamp("us", tz="UTC"), nullable=False),
    pa.field("ee_flow_mw", pa.float32(), nullable=False),
])


class FingridSE1FlowFetcher(FingridSeriesFetcher):
    """
    Fetches the Finland–Northern Sweden (SE1) transmission flow from dataset 60.

    Positive values are exports from Finland; negative values are imports.
    Samples are averaged into the bucket size given by ``FetchRequest.resolution``.
    """

    def __init__(
        self,
        api_key: str,
        dataset_id: int = 60,
        *,
        client: FingridClient | None = None,
        local_path: str | Path | None = DEFAULT_SE1_LOCAL_PATH,
        timeout: float = 30.0,
        max_retries: int = 5,
    ) -> None:
        super().__init__(
            api_key,
            dataset_id=dataset_id,
            dataset_name="se1_flow",
            value_column="se1_flow_mw",
            schema=SE1_FLOW_SCHEMA,
            local_path=local_path,
            client=client,
            timeout=timeout,
            max_retries=max_retries,
        )


class FingridSE3FlowFetcher(FingridSeriesFetcher):
    """
    Fetches the Finland–Central Sweden (SE3) transmission flow from dataset 61.

    Positive values are exports from Finland; negative values are imports.
    Samples are averaged into the bucket size given by ``FetchRequest.resolution``.
    """

    def __init__(
        self,
        api_key: str,
        dataset_id: int = 61,
        *,
        client: FingridClient | None = None,
        local_path: str | Path | None = DEFAULT_SE3_LOCAL_PATH,
        timeout: float = 30.0,
        max_retries: int = 5,
    ) -> None:
        super().__init__(
            api_key,
            dataset_id=dataset_id,
            dataset_name="se3_flow",
            value_column="se3_flow_mw",
            schema=SE3_FLOW_SCHEMA,
            local_path=local_path,
            client=client,
            timeout=timeout,
            max_retries=max_retries,
        )


class FingridEEFlowFetcher(FingridSeriesFetcher):
    """
    Fetches the Finland–Estonia (EE) transmission flow from dataset 55.

    Positive values are exports from Finland; negative values are imports.
    Samples are averaged into the bucket size given by ``FetchRequest.resolution``.
    """

    def __init__(
        self,
        api_key: str,
        dataset_id: int = 55,
        *,
        client: FingridClient | None = None,
        local_path: str | Path | None = DEFAULT_EE_LOCAL_PATH,
        timeout: float = 30.0,
        max_retries: int = 5,
    ) -> None:
        super().__init__(
            api_key,
            dataset_id=dataset_id,
            dataset_name="ee_flow",
            value_column="ee_flow_mw",
            schema=EE_FLOW_SCHEMA,
            local_path=local_path,
            client=client,
            timeout=timeout,
            max_retries=max_retries,
        )