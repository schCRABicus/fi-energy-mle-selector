from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import httpx
import pytest
from tenacity import wait_none

from datasets import FingridClient

API_KEY = "test-api-key"
START = datetime(2024, 1, 1, 0, 0, tzinfo=UTC)
END = datetime(2024, 1, 1, 2, 0, tzinfo=UTC)


def record(timestamp: str, value: Any) -> dict[str, Any]:
    return {"datasetId": 75, "startTime": timestamp, "endTime": timestamp, "value": value}


def payload(records: list[dict[str, Any]], *, last_page: int | None = None) -> dict[str, Any]:
    body: dict[str, Any] = {"data": records}
    if last_page is not None:
        body["pagination"] = {"firstPage": 1, "lastPage": last_page}
    return body


class FakeFingridApi:
    """Scripted Fingrid API served through httpx.MockTransport."""

    def __init__(self, responses: list[Any]) -> None:
        self._responses = list(responses)
        self.requests: list[httpx.Request] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        response = self._responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        if isinstance(response, httpx.Response):
            return response
        return httpx.Response(200, json=response)

    @property
    def urls(self) -> list[httpx.URL]:
        return [request.url for request in self.requests]


def make_client(
    api: FakeFingridApi, *, page_size: int = 20_000, **options: Any
) -> FingridClient:
    return FingridClient(
        API_KEY,
        transport=httpx.MockTransport(api.handler),
        retry_wait=wait_none(),
        page_size=page_size,
        **options,
    )


def samples(client: FingridClient) -> list[tuple[datetime, float | None]]:
    return list(client.iter_samples(75, START, END))


def test_iter_samples_normalizes_timestamps_and_values():
    api = FakeFingridApi([
        payload([
            record("2024-01-01T02:00:00.000+02:00", "12.5"),
            record("2024-01-01T00:15:00.000Z", 3),
        ]),
    ])

    assert samples(make_client(api)) == [
        (datetime(2024, 1, 1, 0, 0, tzinfo=UTC), 12.5),
        (datetime(2024, 1, 1, 0, 15, tzinfo=UTC), 3.0),
    ]


def test_iter_samples_preserves_null_values():
    api = FakeFingridApi([payload([record("2024-01-01T00:00:00Z", None)])])

    assert samples(make_client(api)) == [(datetime(2024, 1, 1, 0, 0, tzinfo=UTC), None)]


def test_iter_samples_follows_pagination_until_last_page():
    api = FakeFingridApi([
        payload([record("2024-01-01T00:00:00Z", 1.0)], last_page=2),
        payload([record("2024-01-01T00:15:00Z", 3.0)], last_page=2),
    ])
    client = make_client(api, page_size=1)

    assert samples(client) == [
        (datetime(2024, 1, 1, 0, 0, tzinfo=UTC), 1.0),
        (datetime(2024, 1, 1, 0, 15, tzinfo=UTC), 3.0),
    ]
    assert len(api.requests) == 2
    assert [url.params["page"] for url in api.urls] == ["1", "2"]


def test_iter_samples_stops_when_unpaginated_page_is_short():
    api = FakeFingridApi([
        payload([
            record("2024-01-01T00:00:00Z", 1.0),
            record("2024-01-01T00:15:00Z", 2.0),
        ]),
        payload([record("2024-01-01T00:30:00Z", 3.0)]),
    ])
    client = make_client(api, page_size=2)

    assert len(samples(client)) == 3
    assert len(api.requests) == 2


def test_iter_samples_accepts_legacy_bare_list_payload():
    api = FakeFingridApi([[record("2024-01-01T00:00:00Z", 4.0)]])
    client = make_client(api, page_size=10)

    assert samples(client) == [(datetime(2024, 1, 1, 0, 0, tzinfo=UTC), 4.0)]
    assert len(api.requests) == 1


def test_client_sends_api_key_and_utc_window():
    api = FakeFingridApi([payload([])])

    samples(make_client(api))

    request = api.requests[0]
    assert request.headers["x-api-key"] == API_KEY
    assert request.url.path == "/api/datasets/75/data"
    assert request.url.params["startTime"] == "2024-01-01T00:00:00Z"
    assert request.url.params["endTime"] == "2024-01-01T02:00:00Z"
    assert request.url.params["pageSize"] == "20000"


def test_client_does_not_retry_client_errors():
    api = FakeFingridApi([httpx.Response(401, json={"error": "invalid api key"})])

    with pytest.raises(httpx.HTTPStatusError) as exc_info:
        samples(make_client(api))

    assert exc_info.value.response.status_code == 401
    assert len(api.requests) == 1


def test_client_retries_rate_limited_requests():
    api = FakeFingridApi([
        httpx.Response(429, json={"message": "Rate limit is exceeded."}),
        payload([record("2024-01-01T00:00:00Z", 4.0)]),
    ])

    assert samples(make_client(api)) == [(datetime(2024, 1, 1, 0, 0, tzinfo=UTC), 4.0)]
    assert len(api.requests) == 2


def test_client_retries_transport_errors():
    api = FakeFingridApi([
        httpx.ConnectError("boom"),
        payload([record("2024-01-01T00:00:00Z", 4.0)]),
    ])

    assert samples(make_client(api)) == [(datetime(2024, 1, 1, 0, 0, tzinfo=UTC), 4.0)]
    assert len(api.requests) == 2


def test_client_gives_up_after_max_retries():
    api = FakeFingridApi([
        httpx.Response(503),
        httpx.Response(503),
        httpx.Response(503),
    ])

    with pytest.raises(httpx.HTTPStatusError) as exc_info:
        samples(make_client(api, max_retries=2))

    assert exc_info.value.response.status_code == 503
    assert len(api.requests) == 3


def test_client_resumes_pagination_after_rate_limit():
    api = FakeFingridApi([
        payload([record("2024-01-01T00:00:00Z", 1.0)], last_page=2),
        httpx.Response(429),
        payload([record("2024-01-01T00:15:00Z", 3.0)], last_page=2),
    ])

    assert len(samples(make_client(api, page_size=1))) == 2
    assert len(api.requests) == 3


def test_client_raises_on_invalid_json():
    api = FakeFingridApi([httpx.Response(200, content=b"<html>rate limited</html>")])

    with pytest.raises(json.JSONDecodeError):
        samples(make_client(api))


def test_client_rejects_record_without_timestamp():
    api = FakeFingridApi([payload([{"value": 1.0}])])

    with pytest.raises(ValueError, match="missing 'startTime'"):
        samples(make_client(api, page_size=10))


def test_client_rejects_missing_api_key():
    with pytest.raises(ValueError, match="API key is required"):
        FingridClient("")


def test_client_rejects_non_positive_page_size():
    with pytest.raises(ValueError, match="page_size must be positive"):
        FingridClient(API_KEY, page_size=0)


def test_client_rejects_negative_max_retries():
    with pytest.raises(ValueError, match="max_retries must be non-negative"):
        FingridClient(API_KEY, max_retries=-1)