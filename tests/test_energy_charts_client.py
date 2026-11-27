from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
import pytest
from tenacity import wait_none

from fi_energy.datasets import EnergyChartsClient

START = datetime(2024, 3, 1, tzinfo=UTC)
END = datetime(2024, 3, 2, tzinfo=UTC)


class FakeEnergyChartsApi:
    """Scripted Energy-Charts responses served through httpx.MockTransport."""

    def __init__(self, payload: Any = None) -> None:
        self._payload = payload
        self.requests: list[httpx.Request] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return httpx.Response(200, json=self._payload)

    @property
    def urls(self) -> list[httpx.URL]:
        return [request.url for request in self.requests]


def make_client(api: FakeEnergyChartsApi, **options: Any) -> EnergyChartsClient:
    return EnergyChartsClient(
        transport=httpx.MockTransport(api.handler),
        retry_wait=wait_none(),
        **options,
    )


def payload(
    entries: list[tuple[datetime, float | None]],
    *,
    unit: str = "EUR / MWh",
) -> dict[str, Any]:
    return {
        "unit": unit,
        "deprecated": False,
        "license_info": "test",
        "unix_seconds": [int(timestamp.timestamp()) for timestamp, _ in entries],
        "price": [value for _, value in entries],
    }


def test_iter_prices_parses_values_and_filters_half_open_window():
    api = FakeEnergyChartsApi(payload([
        (START - timedelta(hours=1), 9.0),
        (START, 10.0),
        (START + timedelta(hours=1), None),
        (END, 11.0),
    ]))

    assert list(make_client(api).iter_prices("FI", START, END)) == [
        (START, 10.0),
        (START + timedelta(hours=1), None),
    ]


def test_client_sends_energy_charts_price_parameters():
    api = FakeEnergyChartsApi(payload([]))
    client = make_client(api)

    list(client.iter_prices("FI", START, END))

    request = api.requests[0]
    assert request.url.path == "/price"
    assert request.url.params["bzn"] == "FI"
    assert request.url.params["start"] == "2024-03-01T00:00:00Z"
    assert request.url.params["end"] == "2024-03-02T00:00:00Z"


def test_client_rejects_unexpected_unit():
    api = FakeEnergyChartsApi(payload([], unit="EUR / kWh"))

    with pytest.raises(ValueError, match="Unexpected Energy-Charts unit"):
        list(make_client(api).iter_prices("FI", START, END))


def test_client_rejects_mismatched_lengths():
    body = payload([(START, 1.0)])
    body["price"].append(2.0)
    api = FakeEnergyChartsApi(body)

    with pytest.raises(ValueError, match="mismatched unix_seconds and price"):
        list(make_client(api).iter_prices("FI", START, END))


@pytest.mark.parametrize(
    "body",
    [
        [],
        {"unit": "EUR / MWh", "price": []},
        {"unit": "EUR / MWh", "unix_seconds": []},
    ],
    ids=["not-an-object", "missing-timestamps", "missing-prices"],
)
def test_client_rejects_malformed_payloads(body):
    api = FakeEnergyChartsApi(body)

    with pytest.raises(ValueError, match="Energy-Charts response"):
        list(make_client(api).iter_prices("FI", START, END))


def test_client_rejects_naive_windows():
    api = FakeEnergyChartsApi(payload([]))

    with pytest.raises(ValueError, match="timezone-aware"):
        list(
            make_client(api).iter_prices(
                "FI", datetime(2024, 3, 1), datetime(2024, 3, 2)
            )
        )