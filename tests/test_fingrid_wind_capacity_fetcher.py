from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from datasets import DataSourceFetcher, FetchRequest, FingridWindCapacityFetcher
from datasets.fingrid.wind_capacity import DEFAULT_LOCAL_PATH

API_KEY = "test-api-key"
START = datetime(2024, 1, 1, 0, 0, tzinfo=UTC)
END = datetime(2024, 1, 1, 2, 0, tzinfo=UTC)
BUNDLED_SAMPLE = (
    Path(__file__).parents[1]
    / "datasets"
    / "fingrid"
    / "data"
    / "wind_capacity_sample.parquet"
)


def write_local(
    tmp_path: Path, rows: list[tuple[datetime, float | None]]
) -> Path:
    """Writes a capacity snapshot with the same layout as the bundled parquet file."""
    schema = pa.schema([
        pa.field("timestamp_utc", pa.timestamp("us", tz="UTC")),
        pa.field("wind_capacity_mw", pa.float32()),
    ])
    table = pa.Table.from_pydict(
        {
            "timestamp_utc": [timestamp for timestamp, _ in rows],
            "wind_capacity_mw": [value for _, value in rows],
        },
        schema=schema,
    )
    path = tmp_path / "wind_capacity.parquet"
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
) -> FingridWindCapacityFetcher:
    return FingridWindCapacityFetcher(
        API_KEY,
        client=client or StubClient(),  # type: ignore[arg-type]
        local_path=local_path,
    )


def test_fetcher_satisfies_datasource_protocol():
    assert isinstance(make_fetcher(None), DataSourceFetcher)


def test_source_name_is_fingrid_268():
    assert make_fetcher(None).source_name == "fingrid_268"


def test_expected_schema_uses_capacity_column():
    assert make_fetcher(None).expected_schema.names == [
        "timestamp_utc",
        "wind_capacity_mw",
    ]


def test_bundled_sample_is_available():
    assert DEFAULT_LOCAL_PATH == BUNDLED_SAMPLE
    assert DEFAULT_LOCAL_PATH.is_file()
    assert DEFAULT_LOCAL_PATH.stat().st_size > 0


def test_fetch_from_local_reads_bundled_sample():
    client = StubClient()
    fetcher = FingridWindCapacityFetcher(API_KEY, client=client)  # type: ignore[arg-type]

    table = fetcher.fetch_from_local(
        FetchRequest(
            datetime(2024, 3, 1, tzinfo=UTC),
            datetime(2024, 3, 2, tzinfo=UTC),
        )
    )

    assert table.schema == fetcher.expected_schema
    assert table.num_rows == 24
    assert table["timestamp_utc"].to_pylist() == [
        datetime(2024, 3, 1, tzinfo=UTC) + timedelta(hours=hour)
        for hour in range(24)
    ]
    assert table["wind_capacity_mw"].to_pylist()[0] == pytest.approx(7027.5)
    assert table["wind_capacity_mw"].to_pylist()[-1] == pytest.approx(7147.5)
    assert client.calls == []


def test_fetch_uses_remote_capacity_dataset():
    client = StubClient([(START, 7000.0)])
    fetcher = make_fetcher(DEFAULT_LOCAL_PATH, client)

    table = fetcher.fetch(FetchRequest(START, END))

    assert table["wind_capacity_mw"].to_pylist() == [pytest.approx(7000.0)]
    assert client.calls == [(268, START, END)]


def test_fetch_from_local_falls_back_when_snapshot_is_missing(tmp_path):
    client = StubClient([(START, 7000.0)])
    fetcher = make_fetcher(tmp_path / "missing.parquet", client)

    table = fetcher.fetch_from_local(FetchRequest(START, END))

    assert table["wind_capacity_mw"].to_pylist() == [pytest.approx(7000.0)]
    assert client.calls == [(268, START, END)]


def test_fetch_from_local_averages_samples_into_requested_buckets(tmp_path):
    rows = [
        (START, 100.0),
        (START + timedelta(minutes=30), 200.0),
    ]
    fetcher = make_fetcher(write_local(tmp_path, rows))

    table = fetcher.fetch_from_local(
        FetchRequest(START, datetime(2024, 1, 1, 1, 0, tzinfo=UTC))
    )

    assert table["wind_capacity_mw"].to_pylist() == [pytest.approx(150.0)]