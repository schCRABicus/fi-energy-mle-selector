from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from fi_energy.datasets import (
    DataSourceFetcher,
    FetchRequest,
    FingridEEFlowFetcher,
    FingridSE1FlowFetcher,
    FingridSE3FlowFetcher,
)
from fi_energy.datasets.fingrid import flows
from fi_energy.datasets.fingrid.series_fetcher import FingridSeriesFetcher

API_KEY = "test-api-key"
START = datetime(2024, 1, 1, 0, 0, tzinfo=UTC)
END = datetime(2024, 1, 1, 2, 0, tzinfo=UTC)


@dataclass(frozen=True)
class FlowCase:
    fetcher_class: type[FingridSeriesFetcher]
    dataset_id: int
    value_column: str
    sample_path: Path
    first_hourly: float
    last_hourly: float
    has_negative: bool


# first/last values are hourly means of the committed 2024-03-01 snapshot
CASES = [
    pytest.param(
        FlowCase(
            FingridSE1FlowFetcher,
            60,
            "se1_flow_mw",
            flows.DEFAULT_SE1_LOCAL_PATH,
            988.3225,
            -529.635,
            has_negative=True,
        ),
        id="se1-60",
    ),
    pytest.param(
        FlowCase(
            FingridSE3FlowFetcher,
            61,
            "se3_flow_mw",
            flows.DEFAULT_SE3_LOCAL_PATH,
            863.91475,
            800.1,
            has_negative=False,
        ),
        id="se3-61",
    ),
    pytest.param(
        FlowCase(
            FingridEEFlowFetcher,
            55,
            "ee_flow_mw",
            flows.DEFAULT_EE_LOCAL_PATH,
            366.3675,
            -23.5,
            has_negative=True,
        ),
        id="ee-55",
    ),
]

DATASET_NAMES = {
    60: "se1_flow",
    61: "se3_flow",
    55: "ee_flow",
}

DEFAULT_LOCAL_PATHS = {
    60: flows.DEFAULT_SE1_LOCAL_PATH,
    61: flows.DEFAULT_SE3_LOCAL_PATH,
    55: flows.DEFAULT_EE_LOCAL_PATH,
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
    case: FlowCase,
    local_path: str | Path | None,
    client: StubClient | None = None,
) -> FingridSeriesFetcher:
    return case.fetcher_class(
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


@pytest.mark.parametrize("case", CASES)
def test_fetcher_satisfies_datasource_protocol(case):
    assert isinstance(make_fetcher(case, None), DataSourceFetcher)


@pytest.mark.parametrize("case", CASES)
def test_source_name_and_schema(case):
    fetcher = make_fetcher(case, None)

    assert fetcher.source_name == f"fingrid_{case.dataset_id}"
    assert fetcher.dataset_name == DATASET_NAMES[case.dataset_id]
    assert fetcher.expected_schema.names == ["timestamp_utc", case.value_column]


@pytest.mark.parametrize("case", CASES)
def test_bundled_sample_is_committed(case):
    assert case.sample_path == DEFAULT_LOCAL_PATHS[case.dataset_id]
    assert case.sample_path.is_file()
    assert case.sample_path.stat().st_size > 0


@pytest.mark.parametrize("case", CASES)
def test_fetch_from_local_resamples_bundled_day_to_hours(case):
    client = StubClient()
    fetcher = case.fetcher_class(
        API_KEY,
        client=client,  # type: ignore[arg-type]
        local_path=case.sample_path,
    )

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

    values = table[case.value_column].to_pylist()
    assert all(math.isfinite(value) for value in values)
    assert any(value > 0 for value in values)
    assert any(value < 0 for value in values) == case.has_negative
    assert values[0] == pytest.approx(case.first_hourly, rel=1e-5)
    assert values[-1] == pytest.approx(case.last_hourly, rel=1e-5)
    assert client.calls == []


@pytest.mark.parametrize("case", CASES)
def test_fetch_from_local_supports_quarter_hour_resolution(case):
    fetcher = make_fetcher(case, case.sample_path)

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


@pytest.mark.parametrize("case", CASES)
def test_fetch_uses_remote_dataset(case):
    client = StubClient([(START, -500.0)])
    fetcher = make_fetcher(case, case.sample_path, client)

    table = fetcher.fetch(FetchRequest(START, END))

    assert table[case.value_column].to_pylist() == [pytest.approx(-500.0)]
    assert client.calls == [(case.dataset_id, START, END)]


@pytest.mark.parametrize("case", CASES)
def test_fetch_from_local_falls_back_when_snapshot_is_missing(case, tmp_path):
    client = StubClient([(START, -500.0)])
    fetcher = make_fetcher(case, tmp_path / "missing.parquet", client)

    table = fetcher.fetch_from_local(FetchRequest(START, END))

    assert table[case.value_column].to_pylist() == [pytest.approx(-500.0)]
    assert client.calls == [(case.dataset_id, START, END)]


@pytest.mark.parametrize("case", CASES)
def test_fetch_from_local_averages_negative_samples(case, tmp_path):
    rows = [
        (START, -100.0),
        (START + timedelta(minutes=30), -200.0),
    ]
    fetcher = make_fetcher(
        case,
        write_local(tmp_path / f"{case.value_column}.parquet", case.value_column, rows),
    )

    table = fetcher.fetch_from_local(
        FetchRequest(START, datetime(2024, 1, 1, 1, 0, tzinfo=UTC))
    )

    assert table[case.value_column].to_pylist() == [pytest.approx(-150.0)]


@pytest.mark.parametrize("case", CASES)
def test_custom_dataset_id_is_used_remotely(case):
    client = StubClient([(START, 1.0)])
    fetcher = case.fetcher_class(
        API_KEY,
        dataset_id=999,
        client=client,  # type: ignore[arg-type]
        local_path=None,
    )

    fetcher.fetch(FetchRequest(START, END))

    assert fetcher.source_name == "fingrid_999"
    assert client.calls == [(999, START, END)]