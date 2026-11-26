from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from datasets import DataSourceFetcher, FetchRequest, NordPoolSpotFetcher
from datasets.energy_charts.spot import DEFAULT_LOCAL_PATHS, SPOT_PRICE_SCHEMA

START = datetime(2024, 1, 1, 0, 0, tzinfo=UTC)
END = datetime(2024, 1, 1, 2, 0, tzinfo=UTC)
_UNSET = object()


class StubClient:
    """Remote client that returns fixed prices and records calls."""

    def __init__(self, prices: list[tuple[datetime, float | None]] | None = None) -> None:
        self._prices = list(prices or [])
        self.calls: list[tuple[str, datetime, datetime]] = []
        self.closed = False

    def iter_prices(
        self, bzn: str, start: datetime, end: datetime
    ) -> list[tuple[datetime, float | None]]:
        self.calls.append((bzn, start, end))
        return list(self._prices)

    def close(self) -> None:
        self.closed = True


def make_fetcher(
    *,
    local_path: str | Path | None | object = _UNSET,
    client: StubClient | None = None,
    bzn: str = "FI",
) -> NordPoolSpotFetcher:
    options: dict[str, Any] = {"client": client or StubClient(), "bzn": bzn}
    if local_path is not _UNSET:
        options["local_path"] = local_path
    return NordPoolSpotFetcher(**options)


def write_local(
    path: Path,
    rows: list[tuple[datetime, float | None]],
    *,
    tz: str | None = "UTC",
) -> Path:
    schema = pa.schema([
        pa.field("timestamp_utc", pa.timestamp("us", tz=tz)),
        pa.field("spot_price_c_kwh", pa.float32()),
    ])
    pq.write_table(
        pa.Table.from_pydict(
            {
                "timestamp_utc": [timestamp for timestamp, _ in rows],
                "spot_price_c_kwh": [value for _, value in rows],
            },
            schema=schema,
        ),
        path,
    )
    return path


def test_fetcher_satisfies_datasource_protocol():
    assert isinstance(make_fetcher(), DataSourceFetcher)


def test_source_name_and_schema():
    fetcher = make_fetcher()

    assert fetcher.source_name == "nordpool_spot_fi"
    assert fetcher.dataset_name == "spot_price_fi"
    assert fetcher.expected_schema == SPOT_PRICE_SCHEMA
    assert fetcher.expected_schema.names == ["timestamp_utc", "spot_price_c_kwh"]
    assert fetcher.expected_schema.field("timestamp_utc").type == pa.timestamp("us", tz="UTC")
    assert fetcher.expected_schema.field("spot_price_c_kwh").type == pa.float32()


def test_non_default_bidding_zone_changes_source_name():
    assert make_fetcher(bzn="se1").source_name == "nordpool_spot_se1"
    assert make_fetcher(bzn="se1").dataset_name == "spot_price_se1"


def test_fetch_converts_eur_per_mwh_to_c_per_kwh():
    client = StubClient([
        (START, 25.0),
        (START + timedelta(hours=1), -0.2),
    ])
    fetcher = make_fetcher(local_path=None, client=client)

    table = fetcher.fetch(FetchRequest(START, END))

    assert table["spot_price_c_kwh"].to_pylist() == [
        pytest.approx(2.5),
        pytest.approx(-0.02),
    ]
    assert table["timestamp_utc"].to_pylist() == [START, START + timedelta(hours=1)]
    assert client.calls == [("FI", START, END)]


def test_fetch_averages_quarter_hourly_prices_into_hours():
    client = StubClient([
        (START + timedelta(minutes=15 * quarter), 10.0 + quarter)
        for quarter in range(4)
    ])
    fetcher = make_fetcher(local_path=None, client=client)

    table = fetcher.fetch(FetchRequest(START, END))

    assert table["timestamp_utc"].to_pylist() == [START]
    assert table["spot_price_c_kwh"].to_pylist() == [pytest.approx(1.15)]


def test_fetch_keeps_missing_prices_as_nulls():
    client = StubClient([(START, None), (START + timedelta(hours=1), 10.0)])
    fetcher = make_fetcher(local_path=None, client=client)

    table = fetcher.fetch(FetchRequest(START, END))

    assert table["spot_price_c_kwh"].to_pylist() == [None, pytest.approx(1.0)]


def test_fetch_filters_rows_outside_the_window():
    client = StubClient([
        (START - timedelta(hours=1), 99.0),
        (START, 10.0),
        (END, 99.0),
    ])
    fetcher = make_fetcher(local_path=None, client=client)

    table = fetcher.fetch(FetchRequest(START, END))

    assert table.num_rows == 1
    assert table["spot_price_c_kwh"].to_pylist() == [pytest.approx(1.0)]


def test_fetch_normalizes_request_timezone_to_utc():
    client = StubClient([(START, 10.0)])
    fetcher = make_fetcher(local_path=None, client=client)
    east_two = timezone(timedelta(hours=2))

    table = fetcher.fetch(
        FetchRequest(
            datetime(2024, 1, 1, 2, 0, tzinfo=east_two),
            datetime(2024, 1, 1, 4, 0, tzinfo=east_two),
        )
    )

    assert table["timestamp_utc"].to_pylist() == [START]
    assert client.calls == [("FI", START, END)]


def test_bundled_sample_is_committed():
    path = DEFAULT_LOCAL_PATHS["FI"]

    assert path.is_file()
    assert path.stat().st_size > 0


def test_fetch_from_local_reads_bundled_day():
    client = StubClient()
    fetcher = make_fetcher(client=client)

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
    values = table["spot_price_c_kwh"].to_pylist()
    assert values[0] == pytest.approx(0.0, abs=1e-5)
    assert values[-1] == pytest.approx(3.117, abs=1e-5)
    assert client.calls == []


def test_fetch_from_local_falls_back_when_snapshot_is_missing(tmp_path):
    client = StubClient([(START, 10.0)])
    fetcher = make_fetcher(
        local_path=tmp_path / "missing.parquet",
        client=client,
    )

    table = fetcher.fetch_from_local(FetchRequest(START, END))

    assert table.num_rows == 1
    assert client.calls == [("FI", START, END)]


def test_fetch_from_local_falls_back_when_disabled():
    client = StubClient([(START, 10.0)])
    fetcher = make_fetcher(local_path=None, client=client)

    table = fetcher.fetch_from_local(FetchRequest(START, END))

    assert table.num_rows == 1
    assert client.calls == [("FI", START, END)]


def test_fetch_from_local_averages_subhourly_values(tmp_path):
    rows = [
        (START, 1.0),
        (START + timedelta(minutes=30), 3.0),
        (START + timedelta(hours=1), 5.0),
    ]
    fetcher = make_fetcher(
        local_path=write_local(tmp_path / "spot.parquet", rows),
    )

    table = fetcher.fetch_from_local(FetchRequest(START, END))

    assert table["spot_price_c_kwh"].to_pylist() == [
        pytest.approx(2.0),
        pytest.approx(5.0),
    ]


def test_fetch_from_local_assumes_utc_for_naive_timestamps(tmp_path):
    rows = [(datetime(2024, 1, 1, 0, 0), 1.0)]
    fetcher = make_fetcher(
        local_path=write_local(tmp_path / "spot.parquet", rows, tz=None),
    )

    table = fetcher.fetch_from_local(FetchRequest(START, END))

    assert table["timestamp_utc"].to_pylist() == [START]
    assert table["spot_price_c_kwh"].to_pylist() == [pytest.approx(1.0)]


def test_fetch_returns_empty_table_for_window_without_data(tmp_path):
    fetcher = make_fetcher(
        local_path=write_local(tmp_path / "spot.parquet", [(START, 1.0)]),
    )

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
            FetchRequest(START, END, resolution="15min"),
            "Unsupported resolution",
        ),
        (
            FetchRequest(END, START),
            "must be before end_utc",
        ),
    ],
)
def test_invalid_requests_are_rejected(request_, message):
    fetcher = make_fetcher(local_path=None)

    with pytest.raises(ValueError, match=message):
        fetcher.fetch(request_)


def test_empty_bidding_zone_is_rejected():
    with pytest.raises(ValueError, match="non-empty bidding-zone"):
        NordPoolSpotFetcher(bzn="  ", client=StubClient())  # type: ignore[arg-type]


def test_context_manager_closes_client():
    client = StubClient()
    fetcher = make_fetcher(local_path=None, client=client)

    with fetcher:
        pass

    assert client.closed