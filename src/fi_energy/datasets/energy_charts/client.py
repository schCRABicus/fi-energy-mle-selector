"""Client for the Energy-Charts open electricity-market API."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from typing import Any

import httpx
from tenacity import RetryCallState

from fi_energy.datasets.http_client import HttpClient
from fi_energy.datasets.utils import utc_window

ENERGY_CHARTS_BASE_URL = "https://api.energy-charts.info/"
PRICE_PATH = "price"
EXPECTED_UNIT = "EUR / MWh"


class EnergyChartsClient(HttpClient):
    """
    Energy-Charts API client for day-ahead electricity prices.

    Energy-Charts publishes Nord Pool / ENTSO-E day-ahead prices for European
    bidding zones. Prices are returned in EUR/MWh, excluding VAT.
    """

    def __init__(
        self,
        *,
        timeout: float = 60.0,
        max_retries: int = 5,
        transport: httpx.BaseTransport | None = None,
        retry_wait: Callable[[RetryCallState], float] | None = None,
    ) -> None:
        super().__init__(
            ENERGY_CHARTS_BASE_URL,
            headers={"User-Agent": "fi-energy-mle-selector/0.1"},
            timeout=timeout,
            max_retries=max_retries,
            transport=transport,
            retry_wait=retry_wait,
        )

    def iter_prices(
        self, bzn: str, start: datetime, end: datetime
    ) -> Iterator[tuple[datetime, float | None]]:
        """
        Yield day-ahead prices as ``(timestamp_utc, EUR/MWh)``.

        Timestamps keep the API's market time-unit cadence: hourly for most of
        the series, and 15-minute from the Finnish 15-minute market switch
        (2025-10-01). The window is half-open ``[start, end)`` in UTC. A missing
        price is yielded as ``None``.
        """
        start, end = utc_window(start, end)
        payload = self.get_json(
            PRICE_PATH,
            params={
                "bzn": bzn,
                "start": _format_timestamp(start),
                "end": _format_timestamp(end),
            },
        )
        yield from _parse_prices(payload, start, end)


def _parse_prices(
    payload: Any, start: datetime, end: datetime
) -> Iterator[tuple[datetime, float | None]]:
    if not isinstance(payload, dict):
        raise ValueError("Energy-Charts response is not a JSON object.")

    unit = payload.get("unit")
    if unit != EXPECTED_UNIT:
        raise ValueError(f"Unexpected Energy-Charts unit: {unit!r}; expected {EXPECTED_UNIT!r}.")

    seconds = payload.get("unix_seconds")
    prices = payload.get("price")
    if not isinstance(seconds, list) or not isinstance(prices, list):
        raise ValueError("Energy-Charts response is missing unix_seconds or price.")
    if len(seconds) != len(prices):
        raise ValueError(
            "Energy-Charts response has mismatched unix_seconds and price lengths."
        )

    for raw_timestamp, raw_price in zip(seconds, prices, strict=True):
        timestamp = datetime.fromtimestamp(int(raw_timestamp), UTC)
        if start <= timestamp < end:
            yield timestamp, None if raw_price is None else float(raw_price)


def _format_timestamp(timestamp: datetime) -> str:
    return timestamp.astimezone(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")