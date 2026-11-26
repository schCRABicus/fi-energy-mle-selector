"""Nord Pool day-ahead spot prices for the Finnish bidding zone."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Self

import pyarrow as pa

from datasets.api import FetchRequest
from datasets.energy_charts.client import EnergyChartsClient
from datasets.utils import iter_local_samples, mean_by_bucket, resolution_step, utc_window

DATA_DIR = Path(__file__).parent / "data"

SPOT_PRICE_SCHEMA = pa.schema([
    pa.field("timestamp_utc", pa.timestamp("us", tz="UTC"), nullable=False),
    pa.field("spot_price_c_kwh", pa.float32()),
])

VALUE_COLUMN = "spot_price_c_kwh"

DEFAULT_LOCAL_PATHS = {
    "FI": DATA_DIR / "fi_spot_sample.parquet",
}

_DEFAULT_LOCAL_PATH = object()


class NordPoolSpotFetcher:
    """
    Fetches hourly Nord Pool day-ahead spot prices in ``c/kWh``.

    Prices come from the Energy-Charts API (which mirrors the ENTSO-E / Nord
    Pool day-ahead series) and are converted from EUR/MWh. Values exclude
    Finnish VAT; a consumer spot contract adds VAT and the retailer margin.
    """

    RESOLUTIONS: dict[str, timedelta] = {"1h": timedelta(hours=1)}

    def __init__(
        self,
        *,
        bzn: str = "FI",
        client: EnergyChartsClient | None = None,
        local_path: str | Path | None | object = _DEFAULT_LOCAL_PATH,
        timeout: float = 60.0,
        max_retries: int = 5,
    ) -> None:
        self._bzn = bzn.strip().upper()
        if not self._bzn:
            raise ValueError("bzn must be a non-empty bidding-zone code.")
        if local_path is _DEFAULT_LOCAL_PATH:
            local_path = DEFAULT_LOCAL_PATHS.get(self._bzn)
        self._local_path = None if local_path is None else Path(local_path)
        self._client = client or EnergyChartsClient(timeout=timeout, max_retries=max_retries)

    @property
    def source_name(self) -> str:
        return f"nordpool_spot_{self._bzn.lower()}"

    @property
    def dataset_name(self) -> str:
        return f"spot_price_{self._bzn.lower()}"

    @property
    def expected_schema(self) -> pa.Schema:
        return SPOT_PRICE_SCHEMA

    def fetch(self, request: FetchRequest) -> pa.Table:
        start, end = utc_window(request.start_utc, request.end_utc)
        resolution_step(self.RESOLUTIONS, request.resolution)
        return self._build_table(self._remote_rows(start, end))

    def fetch_from_local(self, request: FetchRequest) -> pa.Table:
        start, end = utc_window(request.start_utc, request.end_utc)
        step = resolution_step(self.RESOLUTIONS, request.resolution)

        local_path = self._local_path
        if local_path is None or not local_path.exists():
            return self.fetch(request)

        samples = mean_by_bucket(
            iter_local_samples(local_path, value_column=VALUE_COLUMN),
            start=start,
            end=end,
            step=step,
        )
        return self._build_table([
            {"timestamp_utc": timestamp, VALUE_COLUMN: value}
            for timestamp, value in samples
        ])

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def _remote_rows(self, start: datetime, end: datetime) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for timestamp, eur_per_mwh in self._client.iter_prices(self._bzn, start, end):
            if start <= timestamp < end:
                rows.append({
                    "timestamp_utc": timestamp,
                    VALUE_COLUMN: (
                        None if eur_per_mwh is None else _to_c_per_kwh(eur_per_mwh)
                    ),
                })
        return rows

    def _build_table(self, rows: list[dict[str, object]]) -> pa.Table:
        return pa.Table.from_pydict(
            {
                "timestamp_utc": [row["timestamp_utc"] for row in rows],
                VALUE_COLUMN: [row.get(VALUE_COLUMN) for row in rows],
            },
            schema=SPOT_PRICE_SCHEMA,
        )


def _to_c_per_kwh(eur_per_mwh: float) -> float:
    """Converts 1 EUR/MWh = 100 c / 1,000 kWh = 0.1 c/kWh."""
    return eur_per_mwh / 10.0