from __future__ import annotations

import io
import json
import urllib.error
import urllib.parse
from datetime import UTC, datetime
from typing import Any

import pyarrow as pa
import pytest

from datasets import DataSourceFetcher, FetchRequest
from datasets.fingrid_wind import FingridWindFetcher, run_fetcher

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


class FakeResponse:
    def __init__(self, body: bytes) -> None:
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *exc_info: object) -> bool:
        return False


class FakeOpener:
    """Stand-in for urllib.request.urlopen that replays canned responses."""

    def __init__(self, responses: list[Any]) -> None:
        self._responses = list(responses)
        self.requests: list[tuple[Any, float | None]] = []

    def __call__(self, request: Any, timeout: float | None = None) -> FakeResponse:
        self.requests.append((request, timeout))
        response = self._responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        if isinstance(response, bytes):
            return FakeResponse(response)
        return FakeResponse(json.dumps(response).encode())

    @property
    def urls(self) -> list[str]:
        return [request.full_url for request, _ in self.requests]


def make_fetcher(opener: FakeOpener, **kwargs: Any) -> FingridWindFetcher:
    return FingridWindFetcher(API_KEY, opener=opener, **kwargs)


def test_fetcher_satisfies_data_source_protocol():
    fetcher = make_fetcher(FakeOpener([]))

    assert isinstance(fetcher, DataSourceFetcher)
    assert fetcher.source_name == "fingrid_75"


def test_source_name_reflects_custom_dataset_id():
    fetcher = make_fetcher(FakeOpener([]), dataset_id=268)

    assert fetcher.source_name == "fingrid_268"


def test_fetch_resamples_quarter_hourly_samples_to_hourly_means():
    opener = FakeOpener([
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
    fetcher = make_fetcher(opener)

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
    opener = FakeOpener([
        payload([
            record("2024-01-01T00:00:00Z", 1.0),
            record("2024-01-01T00:15:00Z", 2.0),
            record("2024-01-01T00:30:00Z", 3.0),
            record("2024-01-01T00:45:00Z", 4.0),
        ]),
    ])
    fetcher = make_fetcher(opener)

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
    opener = FakeOpener([
        payload([
            record("2024-01-01T00:00:00Z", 100.0),  # before start
            record("2024-01-01T00:15:00Z", 2.0),
            record("2024-01-01T00:45:00Z", 4.0),
            record("2024-01-01T01:00:00Z", 100.0),  # at exclusive end
        ]),
    ])
    fetcher = make_fetcher(opener)

    table = fetcher.fetch(FetchRequest(start_utc=start, end_utc=end))

    assert table.column("timestamp_utc").to_pylist() == [datetime(2024, 1, 1, 0, 0, tzinfo=UTC)]
    assert table.column("wind_mw").to_pylist() == pytest.approx([3.0])


def test_fetch_skips_null_values():
    opener = FakeOpener([
        payload([
            record("2024-01-01T00:00:00Z", None),
            record("2024-01-01T00:15:00Z", 1.0),
            record("2024-01-01T00:30:00Z", 3.0),
        ]),
    ])
    fetcher = make_fetcher(opener)

    table = fetcher.fetch(FetchRequest(start_utc=START, end_utc=END))

    assert table.column("wind_mw").to_pylist() == pytest.approx([2.0])


def test_fetch_averages_duplicate_timestamps():
    opener = FakeOpener([
        payload([
            record("2024-01-01T00:00:00Z", 1.0),
            record("2024-01-01T00:00:00Z", 3.0),
        ]),
    ])
    fetcher = make_fetcher(opener)

    table = fetcher.fetch(FetchRequest(start_utc=START, end_utc=END))

    assert table.column("wind_mw").to_pylist() == pytest.approx([2.0])


def test_fetch_accepts_numeric_strings():
    opener = FakeOpener([payload([record("2024-01-01T00:00:00Z", "12.5")])])
    fetcher = make_fetcher(opener)

    table = fetcher.fetch(FetchRequest(start_utc=START, end_utc=END))

    assert table.column("wind_mw").to_pylist() == pytest.approx([12.5])


def test_fetch_returns_empty_table_with_expected_schema():
    opener = FakeOpener([payload([])])
    fetcher = make_fetcher(opener)

    table = fetcher.fetch(FetchRequest(start_utc=START, end_utc=END))

    assert table.num_rows == 0
    assert table.schema.equals(fetcher.expected_schema, check_metadata=False)


def test_fetch_follows_pagination_until_last_page():
    opener = FakeOpener([
        payload([record("2024-01-01T00:00:00Z", 1.0)], last_page=2),
        payload([record("2024-01-01T00:15:00Z", 3.0)], last_page=2),
    ])
    fetcher = make_fetcher(opener, page_size=1)

    table = fetcher.fetch(FetchRequest(start_utc=START, end_utc=END))

    assert len(opener.requests) == 2
    assert urllib.parse.parse_qs(urllib.parse.urlparse(opener.urls[0]).query)["page"] == ["1"]
    assert urllib.parse.parse_qs(urllib.parse.urlparse(opener.urls[1]).query)["page"] == ["2"]
    assert table.column("wind_mw").to_pylist() == pytest.approx([2.0])


def test_fetch_stops_when_unpaginated_page_is_short():
    opener = FakeOpener([
        payload([
            record("2024-01-01T00:00:00Z", 1.0),
            record("2024-01-01T00:15:00Z", 2.0),
        ]),
        payload([record("2024-01-01T00:30:00Z", 3.0)]),
    ])
    fetcher = make_fetcher(opener, page_size=2)

    table = fetcher.fetch(FetchRequest(start_utc=START, end_utc=END))

    assert len(opener.requests) == 2
    assert table.column("wind_mw").to_pylist() == pytest.approx([2.0])


def test_fetch_accepts_legacy_bare_list_payload():
    opener = FakeOpener([[record("2024-01-01T00:00:00Z", 4.0)]])
    fetcher = make_fetcher(opener, page_size=10)

    table = fetcher.fetch(FetchRequest(start_utc=START, end_utc=END))

    assert table.column("wind_mw").to_pylist() == pytest.approx([4.0])
    assert len(opener.requests) == 1


def test_fetch_sends_api_key_and_utc_window():
    opener = FakeOpener([payload([])])
    fetcher = make_fetcher(opener)

    fetcher.fetch(FetchRequest(start_utc=START, end_utc=END))

    request, timeout = opener.requests[0]
    assert request.get_header("X-api-key") == API_KEY
    assert request.get_header("Accept") == "application/json"
    assert timeout == 30.0
    query = urllib.parse.parse_qs(urllib.parse.urlparse(request.full_url).query)
    assert query["startTime"] == ["2024-01-01T00:00:00Z"]
    assert query["endTime"] == ["2024-01-01T02:00:00Z"]
    assert query["pageSize"] == ["20000"]
    assert "/datasets/75/data" in request.full_url


def http_error(
    code: int, body: bytes = b"", headers: dict[str, str] | None = None
) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        "https://data.fingrid.fi", code, "error", headers, io.BytesIO(body)
    )


def test_fetch_wraps_http_errors_without_retrying_client_errors():
    opener = FakeOpener([http_error(401, b'{"error":"invalid api key"}')])
    fetcher = make_fetcher(opener, sleep=lambda _: pytest.fail("must not sleep on 401"))

    with pytest.raises(RuntimeError, match="HTTP 401.*invalid api key"):
        fetcher.fetch(FetchRequest(start_utc=START, end_utc=END))

    assert len(opener.requests) == 1


def test_fetch_retries_rate_limited_requests():
    sleeps: list[float] = []
    opener = FakeOpener([
        http_error(429, b'{"message":"Rate limit is exceeded. Try again in 2 seconds."}'),
        payload([record("2024-01-01T00:00:00Z", 4.0)]),
    ])
    fetcher = make_fetcher(opener, sleep=sleeps.append)

    table = fetcher.fetch(FetchRequest(start_utc=START, end_utc=END))

    assert sleeps == [2.0]
    assert len(opener.requests) == 2
    assert table.column("wind_mw").to_pylist() == pytest.approx([4.0])


def test_fetch_honours_retry_after_header():
    sleeps: list[float] = []
    opener = FakeOpener([
        http_error(429, headers={"Retry-After": "7"}),
        payload([record("2024-01-01T00:00:00Z", 4.0)]),
    ])
    fetcher = make_fetcher(opener, sleep=sleeps.append)

    fetcher.fetch(FetchRequest(start_utc=START, end_utc=END))

    assert sleeps == [7.0]


def test_fetch_backs_off_exponentially_and_gives_up_after_max_retries():
    sleeps: list[float] = []
    error = http_error(503)
    opener = FakeOpener([error, error, error])
    fetcher = make_fetcher(opener, max_retries=2, sleep=sleeps.append)

    with pytest.raises(RuntimeError, match="HTTP 503"):
        fetcher.fetch(FetchRequest(start_utc=START, end_utc=END))

    assert sleeps == [2.0, 4.0]
    assert len(opener.requests) == 3


def test_fetch_resumes_pagination_after_rate_limit():
    opener = FakeOpener([
        payload([record("2024-01-01T00:00:00Z", 1.0)], last_page=2),
        http_error(429),
        payload([record("2024-01-01T00:15:00Z", 3.0)], last_page=2),
    ])
    fetcher = make_fetcher(opener, page_size=1, sleep=lambda _: None)

    table = fetcher.fetch(FetchRequest(start_utc=START, end_utc=END))

    assert len(opener.requests) == 3
    assert table.column("wind_mw").to_pylist() == pytest.approx([2.0])


def test_fetch_wraps_connection_errors():
    fetcher = make_fetcher(FakeOpener([urllib.error.URLError("boom")]))

    with pytest.raises(RuntimeError, match="Could not reach Fingrid API: boom"):
        fetcher.fetch(FetchRequest(start_utc=START, end_utc=END))


def test_fetch_wraps_invalid_json():
    fetcher = make_fetcher(FakeOpener([b"<html>rate limited</html>"]))

    with pytest.raises(RuntimeError, match="invalid JSON"):
        fetcher.fetch(FetchRequest(start_utc=START, end_utc=END))


def test_fetch_rejects_naive_datetimes():
    fetcher = make_fetcher(FakeOpener([]))

    with pytest.raises(ValueError, match="must be timezone-aware"):
        fetcher.fetch(FetchRequest(start_utc=datetime(2024, 1, 1), end_utc=END))


def test_fetch_rejects_inverted_window():
    fetcher = make_fetcher(FakeOpener([]))

    with pytest.raises(ValueError, match="must be before end_utc"):
        fetcher.fetch(FetchRequest(start_utc=END, end_utc=START))


def test_fetch_rejects_unsupported_resolution():
    fetcher = make_fetcher(FakeOpener([]))

    with pytest.raises(ValueError, match="Unsupported resolution '5min'"):
        fetcher.fetch(FetchRequest(start_utc=START, end_utc=END, resolution="5min"))


def test_constructor_rejects_missing_api_key():
    with pytest.raises(ValueError, match="API key is required"):
        FingridWindFetcher("")


def test_constructor_rejects_non_positive_page_size():
    with pytest.raises(ValueError, match="page_size must be positive"):
        FingridWindFetcher(API_KEY, page_size=0)


def test_constructor_rejects_negative_max_retries():
    with pytest.raises(ValueError, match="max_retries must be non-negative"):
        FingridWindFetcher(API_KEY, max_retries=-1)


def test_fetch_rejects_record_without_timestamp():
    opener = FakeOpener([payload([{"value": 1.0}])])
    fetcher = make_fetcher(opener, page_size=10)

    with pytest.raises(ValueError, match="missing 'startTime'"):
        fetcher.fetch(FetchRequest(start_utc=START, end_utc=END))


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
