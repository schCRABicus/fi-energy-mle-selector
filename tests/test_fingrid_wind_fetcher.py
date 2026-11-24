from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from datasets import DataSourceFetcher, FetchRequest, FingridWindFetcher
from datasets.fingrid_wind.fingrid_wind_fetcher import DEFAULT_LOCAL_PATH

API_KEY = "test-api-key"
START = datetime(2024, 1, 1, 0, 0, tzinfo=UTC)
END = datetime(2024, 1, 1, 2, 0, tzinfo=UTC)
BUNDLED_SAMPLE = Path(__file__).parents[1] / "datasets" / "fingrid_wind" / "data" / "fingrid_wind_sample.parquet"


def write_local(
    tmp_path: Path,
    rows: list[tuple[datetime, float | None]],
    *,
    tz: str | None = "UTC",
) -> Path:
    """Writes a local snapshot with the same layout as the bundled parquet file."""
    schema = pa.schema([
        pa.field("timestamp_utc", pa.timestamp("us", tz=tz)),
        pa.field("wind_mw", pa.float32()),
    ])
    table = pa.Table.from_pydict(
        {
            "timestamp_utc": [timestamp for timestamp, _ in rows],
            "wind_mw": [value for _, value in rows],
        },
        schema=schema,
    )
    path = tmp_path / "wind.parquet"
    pq.write_table(table, path)
    return path


class StubClient:
    """Remote client that returns a fixed sample set and records calls."""

    def __init__(self, samples: list[tuple[datetime, float | None]] | None = None) -> None:
        self._samples = list(samples or [])
        self.calls: list[tuple[int, datetime, datetime]] = []

    def iter_samples(
        self, dataset_id: int, start: datetime, end: datetime
    ) -> list[tuple[datetime, float | None]]:
        self.calls.append((dataset_id, start, end))
        return list(self._samples)

    def close(self) -> None:
        pass


def make_fetcher(
    local_path: str | Path | None,
    client: StubClient | None = None,
) -> FingridWindFetcher:
    return FingridWindFetcher(
        API_KEY,
        client=client or StubClient(),  # type: ignore[arg-type]
        local_path=local_path,
    )


def test_fetcher_satisfies_datasource_protocol():
    assert isinstance(make_fetcher(None), DataSourceFetcher)


def test_source_name_uses_dataset_id():
    assert make_fetcher(None).source_name == "fingrid_75"
    assert (
        FingridWindFetcher(API_KEY, dataset_id=245, client=StubClient(), local_path=None).source_name
        == "fingrid_245"
    )


def test_bundled_sample_is_available():
    assert DEFAULT_LOCAL_PATH == BUNDLED_SAMPLE
    assert DEFAULT_LOCAL_PATH.is_file()
    assert DEFAULT_LOCAL_PATH.stat().st_size > 0


def test_fetch_from_local_reads_bundled_sample():
    client = StubClient()
    fetcher = FingridWindFetcher(API_KEY, client=client)  # type: ignore[arg-type]

    table = fetcher.fetch_from_local(FetchRequest(START, datetime(2024, 3, 1, 1, 0, tzinfo=UTC)))

    assert table.schema == fetcher.expected_schema
    assert table.num_rows == 1
    assert table["timestamp_utc"].to_pylist() == [datetime(2024, 3, 1, 0, 0, tzinfo=UTC)]
    assert table["wind_mw"].to_pylist() == [pytest.approx(5531.6051, rel=1e-4)]
    assert client.calls == []


def test_fetch_from_local_resamples_bundled_day():
    fetcher = make_fetcher(DEFAULT_LOCAL_PATH)

    table = fetcher.fetch_from_local(
        FetchRequest(
            datetime(2024, 3, 1, tzinfo=UTC),
            datetime(2024, 3, 2, tzinfo=UTC),
        )
    )

    assert table.num_rows == 24
    assert table["timestamp_utc"].to_pylist()[0] == datetime(2024, 3, 1, tzinfo=UTC)
    assert table["timestamp_utc"].to_pylist()[-1] == datetime(2024, 3, 1, 23, tzinfo=UTC)
    assert all(value > 0 for value in table["wind_mw"].to_pylist())


def test_fetch_uses_remote_even_when_local_snapshot_exists():
    client = StubClient([(START, 4.0)])
    fetcher = make_fetcher(DEFAULT_LOCAL_PATH, client)

    table = fetcher.fetch(FetchRequest(START, END))

    assert table["wind_mw"].to_pylist() == [pytest.approx(4.0)]
    assert len(client.calls) == 1


def test_fetch_from_local_falls_back_when_snapshot_is_missing(tmp_path):
    client = StubClient([(START, 4.0)])
    fetcher = make_fetcher(tmp_path / "missing.parquet", client)

    table = fetcher.fetch_from_local(FetchRequest(START, END))

    assert table["wind_mw"].to_pylist() == [pytest.approx(4.0)]
    assert client.calls == [(75, START, END)]


def test_fetch_from_local_falls_back_when_disabled():
    client = StubClient([(START, 4.0)])
    fetcher = make_fetcher(None, client)

    table = fetcher.fetch_from_local(FetchRequest(START, END))

    assert table["wind_mw"].to_pylist() == [pytest.approx(4.0)]
    assert len(client.calls) == 1


def test_fetch_from_local_averages_quarter_hour_samples_into_hours(tmp_path):
    rows = [(START + timedelta(minutes=15 * i), float(i + 1)) for i in range(8)]
    fetcher = make_fetcher(write_local(tmp_path, rows))

    table = fetcher.fetch_from_local(FetchRequest(START, END))

    assert table["wind_mw"].to_pylist() == [pytest.approx(2.5), pytest.approx(6.5)]


def test_fetch_from_local_supports_fifteen_minute_resolution(tmp_path):
    rows = [(START.replace(minute=15 * i), float(i + 1)) for i in range(4)]
    fetcher = make_fetcher(write_local(tmp_path, rows))

    table = fetcher.fetch_from_local(
        FetchRequest(
            START,
            datetime(2024, 1, 1, 1, 0, tzinfo=UTC),
            resolution="15min",
        )
    )

    assert table["wind_mw"].to_pylist() == [
        pytest.approx(1.0),
        pytest.approx(2.0),
        pytest.approx(3.0),
        pytest.approx(4.0),
    ]


def test_fetch_from_local_assumes_utc_for_naive_timestamps(tmp_path):
    rows = [(datetime(2024, 1, 1, 0, 0), 1.0)]
    fetcher = make_fetcher(write_local(tmp_path, rows, tz=None))

    table = fetcher.fetch_from_local(FetchRequest(START, END))

    assert table["timestamp_utc"].to_pylist() == [START]
    assert table["wind_mw"].to_pylist() == [pytest.approx(1.0)]


def test_fetch_from_local_normalizes_request_timezone_to_utc():
    fetcher = make_fetcher(BUNDLED_SAMPLE)
    east_two = timezone(timedelta(hours=2))

    table = fetcher.fetch_from_local(
        FetchRequest(
            datetime(2024, 3, 1, 5, 0, tzinfo=east_two),
            datetime(2024, 3, 1, 7, 0, tzinfo=east_two),
        )
    )

    assert table.num_rows == 2
    assert table["timestamp_utc"].to_pylist() == [
        datetime(2024, 3, 1, 3, 0, tzinfo=UTC),
        datetime(2024, 3, 1, 4, 0, tzinfo=UTC),
    ]


def test_fetch_uses_half_open_window(tmp_path):
    rows = [
        (START, 100.0),
        (START.replace(minute=15), 2.0),
        (START.replace(minute=45), 4.0),
        (datetime(2024, 1, 1, 1, 0, tzinfo=UTC), 100.0),
    ]
    fetcher = make_fetcher(write_local(tmp_path, rows))

    table = fetcher.fetch_from_local(
        FetchRequest(START.replace(minute=15), datetime(2024, 1, 1, 1, 0, tzinfo=UTC))
    )

    assert table["wind_mw"].to_pylist() == [pytest.approx(3.0)]


def test_fetch_skips_null_values(tmp_path):
    rows = [
        (START, None),
        (START.replace(minute=15), 1.0),
        (START.replace(minute=30), 3.0),
    ]
    fetcher = make_fetcher(write_local(tmp_path, rows))

    table = fetcher.fetch_from_local(
        FetchRequest(START, datetime(2024, 1, 1, 1, 0, tzinfo=UTC))
    )

    assert table["wind_mw"].to_pylist() == [pytest.approx(2.0)]


def test_fetch_averages_duplicate_timestamps(tmp_path):
    rows = [(START, 1.0), (START, 3.0)]
    fetcher = make_fetcher(write_local(tmp_path, rows))

    table = fetcher.fetch_from_local(
        FetchRequest(START, datetime(2024, 1, 1, 1, 0, tzinfo=UTC))
    )

    assert table["wind_mw"].to_pylist() == [pytest.approx(2.0)]


def test_fetch_returns_empty_table_for_window_without_data(tmp_path):
    fetcher = make_fetcher(write_local(tmp_path, [(START, 1.0)]))

    table = fetcher.fetch_from_local(
        FetchRequest(
            datetime(2024, 1, 1, 5, 0, tzinfo=UTC),
            datetime(2024, 1, 1, 6, 0, tzinfo=UTC),
        )
    )

    assert table.num_rows == 0
    assert table.schema == fetcher.expected_schema


@pytest.mark.parametrize(
    ("request_", "message"),
    [
        (
            FetchRequest(datetime(2024, 1, 1), datetime(2024, 1, 2)),
            "timezone-aware",
        ),
        (
            FetchRequest(START, END, resolution="1d"),
            "Unsupported resolution",
        ),
        (
            FetchRequest(END, START),
            "must be before end_utc",
        ),
    ],
)
def test_invalid_requests_are_rejected(request_, message):
    fetcher = make_fetcher(BUNDLED_SAMPLE)

    with pytest.raises(ValueError, match=message):
        fetcher.fetch_from_local(request_)


def test_context_manager_closes_client():
    client = StubClient()
    fetcher = make_fetcher(None, client)
    closed = False

    def close() -> None:
        nonlocal closed
        closed = True

    client.close = close  # type: ignore[method-assign]
    with fetcher:
        pass

    assert closed