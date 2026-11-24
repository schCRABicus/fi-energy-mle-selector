from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import httpx
import pyarrow as pa
import pytest
from tenacity import wait_none

from datasets import (
    DataSourceFetcher,
    FetchRequest,
    FingridClient,
    FingridWindFetcher,
)
from datasets.fingrid_wind import run_fetcher

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


def make_fetcher(
    api: FakeFingridApi,
    *,
    dataset_id: int = 75,
    page_size: int = 20_000,
    **client_options: Any,
) -> FingridWindFetcher:
    client = FingridClient(
        API_KEY,
        transport=httpx.MockTransport(api.handler),
        retry_wait=wait_none(),
        page_size=page_size,
        **client_options,
    )
    return FingridWindFetcher(API_KEY, dataset_id=dataset_id, client=client)


def test_fetcher_satisfies_data_source_protocol():
    fetcher = make_fetcher(FakeFingridApi([]))

    assert isinstance(fetcher, DataSourceFetcher)
    assert fetcher.source_name == "fingrid_75"


def test_source_name_reflects_custom_dataset_id():
    fetcher = make_fetcher(FakeFingridApi([]), dataset_id=268)

    assert fetcher.source_name == "fingrid_268"


def test_fetch_resamples_quarter_hourly_samples_to_hourly_means():
    api = FakeFingridApi([
        payload([
            record("2024-01-01T00:00:00.000Z", 1.0),
            record("2024-01-01T00:15:00.000Z", 2.0),
            record("2024-01-01T00:30:00.000Z", 3.0),
            record("2024-01-01T00:45:00.000Z", 4.0),
            record("2024-01-01T01:00:00.000Z", 5.0),
            record("2024-01-01T01:15:00.000Z", 6.0),
            record("2024-01-01T01:30:00.000Z", 7.0),
            record("2024-01-01T01:45:00.000Z", 8.0),
        ]),
    ])
    fetcher = make_fetcher(api)

    table = fetcher.fetch(FetchRequest(start_utc=START, end_utc=END))

    assert table.schema.equals(fetcher.expected_schema, check_metadata=False)
    assert table.column("timestamp_utc").to_pylist() == [
        datetime(2024, 1, 1, 0, 0, tzinfo=UTC),
        datetime(2024, 1, 1, 1, 0, tzinfo=UTC),
    ]
    assert table.column("wind_mw").to_pylist() == pytest.approx([2.5, 6.5])
    assert table.column("timestamp_utc").type == pa.timestamp("us", tz="UTC")
    assert table.column("wind_mw").type == pa.float32()


def test_fetch_honours_finer_requested_resolution():
    api = FakeFingridApi([
        payload([
            record("2024-01-01T00:00:00Z", 1.0),
            record("2024-01-01T00:15:00Z", 2.0),
            record("2024-01-01T00:30:00Z", 3.0),
            record("2024-01-01T00:45:00Z", 4.0),
        ]),
    ])
    fetcher = make_fetcher(api)

    table = fetcher.fetch(FetchRequest(start_utc=START, end_utc=END, resolution="15min"))

    assert table.column("timestamp_utc").to_pylist() == [
        datetime(2024, 1, 1, 0, 0, tzinfo=UTC),
        datetime(2024, 1, 1, 0, 15, tzinfo=UTC),
        datetime(2024, 1, 1, 0, 30, tzinfo=UTC),
        datetime(2024, 1, 1, 0, 45, tzinfo=UTC),
    ]
    assert table.column("wind_mw").to_pylist() == pytest.approx([1.0, 2.0, 3.0, 4.0])


def test_fetch_only_includes_rows_inside_half_open_window():
    start = datetime(2024, 1, 1, 0, 15, tzinfo=UTC)
    end = datetime(2024, 1, 1, 1, 0, tzinfo=UTC)
    api = FakeFingridApi([
        payload([
            record("2024-01-01T00:00:00Z", 100.0),  # before start
            record("2024-01-01T00:15:00Z", 2.0),
            record("2024-01-01T00:45:00Z", 4.0),
            record("2024-01-01T01:00:00Z", 100.0),  # at exclusive end
        ]),
    ])
    fetcher = make_fetcher(api)

    table = fetcher.fetch(FetchRequest(start_utc=start, end_utc=end))

    assert table.column("timestamp_utc").to_pylist() == [datetime(2024, 1, 1, 0, 0, tzinfo=UTC)]
    assert table.column("wind_mw").to_pylist() == pytest.approx([3.0])


def test_fetch_skips_null_values():
    api = FakeFingridApi([
        payload([
            record("2024-01-01T00:00:00Z", None),
            record("2024-01-01T00:15:00Z", 1.0),
            record("2024-01-01T00:30:00Z", 3.0),
        ]),
    ])
    fetcher = make_fetcher(api)

    table = fetcher.fetch(FetchRequest(start_utc=START, end_utc=END))

    assert table.column("wind_mw").to_pylist() == pytest.approx([2.0])


def test_fetch_averages_duplicate_timestamps():
    api = FakeFingridApi([
        payload([
            record("2024-01-01T00:00:00Z", 1.0),
            record("2024-01-01T00:00:00Z", 3.0),
        ]),
    ])
    fetcher = make_fetcher(api)

    table = fetcher.fetch(FetchRequest(start_utc=START, end_utc=END))

    assert table.column("wind_mw").to_pylist() == pytest.approx([2.0])


def test_fetch_accepts_numeric_strings():
    api = FakeFingridApi([payload([record("2024-01-01T00:00:00Z", "12.5")])])
    fetcher = make_fetcher(api)

    table = fetcher.fetch(FetchRequest(start_utc=START, end_utc=END))

    assert table.column("wind_mw").to_pylist() == pytest.approx([12.5])


def test_fetch_returns_empty_table_with_expected_schema():
    api = FakeFingridApi([payload([])])
    fetcher = make_fetcher(api)

    table = fetcher.fetch(FetchRequest(start_utc=START, end_utc=END))

    assert table.num_rows == 0
    assert table.schema.equals(fetcher.expected_schema, check_metadata=False)


def test_fetch_follows_pagination_until_last_page():
    api = FakeFingridApi([
        payload([record("2024-01-01T00:00:00Z", 1.0)], last_page=2),
        payload([record("2024-01-01T00:15:00Z", 3.0)], last_page=2),
    ])
    fetcher = make_fetcher(api, page_size=1)

    table = fetcher.fetch(FetchRequest(start_utc=START, end_utc=END))

    assert len(api.requests) == 2
    assert [url.params["page"] for url in api.urls] == ["1", "2"]
    assert table.column("wind_mw").to_pylist() == pytest.approx([2.0])


def test_fetch_stops_when_unpaginated_page_is_short():
    api = FakeFingridApi([
        payload([
            record("2024-01-01T00:00:00Z", 1.0),
            record("2024-01-01T00:15:00Z", 2.0),
        ]),
        payload([record("2024-01-01T00:30:00Z", 3.0)]),
    ])
    fetcher = make_fetcher(api, page_size=2)

    table = fetcher.fetch(FetchRequest(start_utc=START, end_utc=END))

    assert len(api.requests) == 2
    assert table.column("wind_mw").to_pylist() == pytest.approx([2.0])


def test_fetch_accepts_legacy_bare_list_payload():
    api = FakeFingridApi([[record("2024-01-01T00:00:00Z", 4.0)]])
    fetcher = make_fetcher(api, page_size=10)

    table = fetcher.fetch(FetchRequest(start_utc=START, end_utc=END))

    assert table.column("wind_mw").to_pylist() == pytest.approx([4.0])
    assert len(api.requests) == 1


def test_fetch_sends_api_key_and_utc_window():
    api = FakeFingridApi([payload([])])
    fetcher = make_fetcher(api)

    fetcher.fetch(FetchRequest(start_utc=START, end_utc=END))

    request = api.requests[0]
    assert request.headers["x-api-key"] == API_KEY
    assert request.url.path == "/api/datasets/75/data"
    assert request.url.params["startTime"] == "2024-01-01T00:00:00Z"
    assert request.url.params["endTime"] == "2024-01-01T02:00:00Z"
    assert request.url.params["pageSize"] == "20000"


def test_fetch_raises_on_non_retryable_http_error():
    api = FakeFingridApi([httpx.Response(401, json={"error": "invalid api key"})])
    fetcher = make_fetcher(api)

    with pytest.raises(httpx.HTTPStatusError) as exc_info:
        fetcher.fetch(FetchRequest(start_utc=START, end_utc=END))

    assert exc_info.value.response.status_code == 401
    assert len(api.requests) == 1


def test_fetch_retries_rate_limited_requests():
    api = FakeFingridApi([
        httpx.Response(429, json={"message": "Rate limit is exceeded."}),
        payload([record("2024-01-01T00:00:00Z", 4.0)]),
    ])
    fetcher = make_fetcher(api)

    table = fetcher.fetch(FetchRequest(start_utc=START, end_utc=END))

    assert len(api.requests) == 2
    assert table.column("wind_mw").to_pylist() == pytest.approx([4.0])


def test_fetch_retries_transport_errors():
    api = FakeFingridApi([
        httpx.ConnectError("boom"),
        payload([record("2024-01-01T00:00:00Z", 4.0)]),
    ])
    fetcher = make_fetcher(api)

    table = fetcher.fetch(FetchRequest(start_utc=START, end_utc=END))

    assert len(api.requests) == 2
    assert table.column("wind_mw").to_pylist() == pytest.approx([4.0])


def test_fetch_gives_up_after_max_retries():
    api = FakeFingridApi([
        httpx.Response(503),
        httpx.Response(503),
        httpx.Response(503),
    ])
    fetcher = make_fetcher(api, max_retries=2)

    with pytest.raises(httpx.HTTPStatusError) as exc_info:
        fetcher.fetch(FetchRequest(start_utc=START, end_utc=END))

    assert exc_info.value.response.status_code == 503
    assert len(api.requests) == 3


def test_fetch_resumes_pagination_after_rate_limit():
    api = FakeFingridApi([
        payload([record("2024-01-01T00:00:00Z", 1.0)], last_page=2),
        httpx.Response(429),
        payload([record("2024-01-01T00:15:00Z", 3.0)], last_page=2),
    ])
    fetcher = make_fetcher(api, page_size=1)

    table = fetcher.fetch(FetchRequest(start_utc=START, end_utc=END))

    assert len(api.requests) == 3
    assert table.column("wind_mw").to_pylist() == pytest.approx([2.0])


def test_fetch_raises_on_invalid_json():
    api = FakeFingridApi([httpx.Response(200, content=b"<html>rate limited</html>")])
    fetcher = make_fetcher(api)

    with pytest.raises(json.JSONDecodeError):
        fetcher.fetch(FetchRequest(start_utc=START, end_utc=END))


def test_fetch_rejects_naive_datetimes():
    fetcher = make_fetcher(FakeFingridApi([]))

    with pytest.raises(ValueError, match="must be timezone-aware"):
        fetcher.fetch(FetchRequest(start_utc=datetime(2024, 1, 1), end_utc=END))


def test_fetch_rejects_inverted_window():
    fetcher = make_fetcher(FakeFingridApi([]))

    with pytest.raises(ValueError, match="must be before end_utc"):
        fetcher.fetch(FetchRequest(start_utc=END, end_utc=START))


def test_fetch_rejects_unsupported_resolution():
    fetcher = make_fetcher(FakeFingridApi([]))

    with pytest.raises(ValueError, match="Unsupported resolution '5min'"):
        fetcher.fetch(FetchRequest(start_utc=START, end_utc=END, resolution="5min"))


def test_fetch_rejects_record_without_timestamp():
    api = FakeFingridApi([payload([{"value": 1.0}])])
    fetcher = make_fetcher(api, page_size=10)

    with pytest.raises(ValueError, match="missing 'startTime'"):
        fetcher.fetch(FetchRequest(start_utc=START, end_utc=END))


def test_client_rejects_missing_api_key():
    with pytest.raises(ValueError, match="API key is required"):
        FingridClient("")


def test_client_rejects_non_positive_page_size():
    with pytest.raises(ValueError, match="page_size must be positive"):
        FingridClient(API_KEY, page_size=0)


def test_client_rejects_negative_max_retries():
    with pytest.raises(ValueError, match="max_retries must be non-negative"):
        FingridClient(API_KEY, max_retries=-1)


class _StubFetcher:
    source_name = "stub"

    def __init__(self, expected_schema: pa.Schema, table: pa.Table) -> None:
        self.expected_schema = expected_schema
        self._table = table

    def fetch(self, request: FetchRequest) -> pa.Table:
        return self._table


def test_run_fetcher_returns_validated_table():
    schema = pa.schema([pa.field("value", pa.int64())])
    table = pa.table({"value": [1, 2, 3]}, schema=schema)

    result = run_fetcher(
        _StubFetcher(schema, table),
        FetchRequest(start_utc=START, end_utc=END),
    )

    assert result is table


def test_run_fetcher_rejects_schema_mismatch():
    expected = pa.schema([pa.field("value", pa.int64())])
    wrong = pa.table({"value": pa.array([1.0], type=pa.float32())})

    with pytest.raises(TypeError, match="Schema mismatch"):
        run_fetcher(
            _StubFetcher(expected, wrong),
            FetchRequest(start_utc=START, end_utc=END),
        )