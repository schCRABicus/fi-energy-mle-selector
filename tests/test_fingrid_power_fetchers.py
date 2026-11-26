from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from datasets import (
    DataSourceFetcher,
    FetchRequest,
    FingridConsumptionFetcher,
    FingridHydroFetcher,
    FingridNuclearFetcher,
)
from datasets.fingrid import consumption, hydro, nuclear
from datasets.fingrid.series_fetcher import FingridSeriesFetcher

API_KEY = "test-api-key"
START = datetime(2024, 1, 1, 0, 0, tzinfo=UTC)
END = datetime(2024, 1, 1, 2, 0, tzinfo=UTC)
DATA_DIR = Path(__file__).parents[1] / "datasets" / "fingrid" / "data"

# first/last values are hourly means of the committed 2024-03-01 snapshot
CASES = [
    pytest.param(
        FingridConsumptionFetcher,
        124,
        "consumption_mw",
        DATA_DIR / "consumption_sample.parquet",
        10377.575,
        10217.65,
        id="consumption-124",
    ),
    pytest.param(
        FingridNuclearFetcher,
        188,
        "nuclear_mw",
        DATA_DIR / "nuclear_sample.parquet",
        4228.75999875,
        4015.049999,
        id="nuclear-188",
    ),
    pytest.param(
        FingridHydroFetcher,
        191,
        "hydro_mw",
        DATA_DIR / "hydro_sample.parquet",
        996.00599825,
        1288.34999975,
        id="hydro-191",
    ),
]

DATASET_NAMES = {
    124: "electricity_consumption",
    188: "nuclear_production",
    191: "hydro_production",
}


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
    fetcher_class: type[FingridSeriesFetcher],
    local_path: str | Path | None,
    client: StubClient | None = None,
) -> FingridSeriesFetcher:
    return fetcher_class(
        API_KEY,
        client=client or StubClient(),  # type: ignore[arg-type]
        local_path=local_path,
    )


def write_local(
    path: Path,
    value_column: str,
    rows: list[tuple[datetime, float | None]],
) -> Path:
    """Writes a local snapshot with the same layout as a bundled parquet file."""
    schema = pa.schema([
        pa.field("timestamp_utc", pa.timestamp("us", tz="UTC")),
        pa.field(value_column, pa.float32()),
    ])
    table = pa.Table.from_pydict(
        {
            "timestamp_utc": [timestamp for timestamp, _ in rows],
            value_column: [value for _, value in rows],
        },
        schema=schema,
    )
    pq.write_table(table, path)
    return path


@pytest.mark.parametrize(
    ("fetcher_class", "dataset_id", "value_column", "sample_path", "first", "last"),
    CASES,
)
def test_fetcher_satisfies_datasource_protocol(
    fetcher_class, dataset_id, value_column, sample_path, first, last
):
    assert isinstance(make_fetcher(fetcher_class, None), DataSourceFetcher)


@pytest.mark.parametrize(
    ("fetcher_class", "dataset_id", "value_column", "sample_path", "first", "last"),
    CASES,
)
def test_source_name_and_schema(
    fetcher_class, dataset_id, value_column, sample_path, first, last
):
    fetcher = make_fetcher(fetcher_class, None)

    assert fetcher.source_name == f"fingrid_{dataset_id}"
    assert fetcher.dataset_name == DATASET_NAMES[dataset_id]
    assert fetcher.expected_schema.names == ["timestamp_utc", value_column]


@pytest.mark.parametrize(
    ("fetcher_class", "dataset_id", "value_column", "sample_path", "first", "last"),
    CASES,
)
def test_bundled_sample_is_committed(
    fetcher_class, dataset_id, value_column, sample_path, first, last
):
    module = {124: consumption, 188: nuclear, 191: hydro}[dataset_id]

    assert module.DEFAULT_LOCAL_PATH == sample_path
    assert sample_path.is_file()
    assert sample_path.stat().st_size > 0


@pytest.mark.parametrize(
    ("fetcher_class", "dataset_id", "value_column", "sample_path", "first", "last"),
    CASES,
)
def test_fetch_from_local_resamples_bundled_day_to_hours(
    fetcher_class, dataset_id, value_column, sample_path, first, last
):
    client = StubClient()
    fetcher = fetcher_class(API_KEY, client=client, local_path=sample_path)  # type: ignore[arg-type]

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
    assert table[value_column].to_pylist()[0] == pytest.approx(first, rel=1e-5)
    assert table[value_column].to_pylist()[-1] == pytest.approx(last, rel=1e-5)
    assert all(math.isfinite(value) and value > 0 for value in table[value_column].to_pylist())
    assert client.calls == []


@pytest.mark.parametrize(
    ("fetcher_class", "dataset_id", "value_column", "sample_path", "first", "last"),
    CASES,
)
def test_fetch_from_local_supports_quarter_hour_resolution(
    fetcher_class, dataset_id, value_column, sample_path, first, last
):
    fetcher = make_fetcher(fetcher_class, sample_path)

    table = fetcher.fetch_from_local(
        FetchRequest(
            datetime(2024, 3, 1, tzinfo=UTC),
            datetime(2024, 3, 2, tzinfo=UTC),
            resolution="15min",
        )
    )

    assert table.num_rows == 96
    assert {timestamp.minute for timestamp in table["timestamp_utc"].to_pylist()} == {
        0,
        15,
        30,
        45,
    }


@pytest.mark.parametrize(
    ("fetcher_class", "dataset_id", "value_column", "sample_path", "first", "last"),
    CASES,
)
def test_fetch_uses_remote_dataset(
    fetcher_class, dataset_id, value_column, sample_path, first, last
):
    client = StubClient([(START, 7000.0)])
    fetcher = make_fetcher(fetcher_class, sample_path, client)

    table = fetcher.fetch(FetchRequest(START, END))

    assert table[value_column].to_pylist() == [pytest.approx(7000.0)]
    assert client.calls == [(dataset_id, START, END)]


@pytest.mark.parametrize(
    ("fetcher_class", "dataset_id", "value_column", "sample_path", "first", "last"),
    CASES,
)
def test_fetch_from_local_falls_back_when_snapshot_is_missing(
    fetcher_class, dataset_id, value_column, sample_path, first, last, tmp_path
):
    client = StubClient([(START, 7000.0)])
    fetcher = make_fetcher(fetcher_class, tmp_path / "missing.parquet", client)

    table = fetcher.fetch_from_local(FetchRequest(START, END))

    assert table[value_column].to_pylist() == [pytest.approx(7000.0)]
    assert client.calls == [(dataset_id, START, END)]


@pytest.mark.parametrize(
    ("fetcher_class", "dataset_id", "value_column", "sample_path", "first", "last"),
    CASES,
)
def test_fetch_from_local_averages_samples_into_requested_buckets(
    fetcher_class, dataset_id, value_column, sample_path, first, last, tmp_path
):
    rows = [
        (START, 100.0),
        (START + timedelta(minutes=30), 200.0),
    ]
    fetcher = make_fetcher(
        fetcher_class,
        write_local(tmp_path / f"{value_column}.parquet", value_column, rows),
    )

    table = fetcher.fetch_from_local(
        FetchRequest(START, datetime(2024, 1, 1, 1, 0, tzinfo=UTC))
    )

    assert table[value_column].to_pylist() == [pytest.approx(150.0)]


@pytest.mark.parametrize(
    ("fetcher_class", "dataset_id", "value_column", "sample_path", "first", "last"),
    CASES,
)
def test_custom_dataset_id_is_used_remotely(
    fetcher_class, dataset_id, value_column, sample_path, first, last
):
    client = StubClient([(START, 1.0)])
    fetcher = fetcher_class(
        API_KEY,
        dataset_id=999,
        client=client,  # type: ignore[arg-type]
        local_path=None,
    )

    fetcher.fetch(FetchRequest(START, END))

    assert fetcher.source_name == "fingrid_999"
    assert client.calls == [(999, START, END)]


def test_invalid_resolution_is_rejected():
    fetcher = make_fetcher(FingridConsumptionFetcher, consumption.DEFAULT_LOCAL_PATH)

    with pytest.raises(ValueError, match="Unsupported resolution"):
        fetcher.fetch_from_local(FetchRequest(START, END, resolution="3min"))