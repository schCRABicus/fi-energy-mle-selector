from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from datasets import DataSourceFetcher, FetchRequest, FmiWeatherFetcher
from datasets.fmi.weather import (
    DEFAULT_LOCAL_PATHS,
    STATIONS,
    WEATHER_SCHEMA,
    station_for_city,
)

START = datetime(2024, 1, 1, 0, 0, tzinfo=UTC)
END = datetime(2024, 1, 1, 2, 0, tzinfo=UTC)
_UNSET = object()

CITY_CASES = [
    pytest.param(
        "helsinki",
        {
            "temperature_c": (0.3, 0.3),
            "wind_speed_ms": (4.2, 4.5),
            "radiation_wm2": (-0.5, -0.3),
        },
        id="helsinki",
    ),
    pytest.param(
        "vaasa",
        {
            "temperature_c": (1.2, 2.0),
            "wind_speed_ms": (7.4, 5.5),
            "radiation_wm2": (0.0, 0.0),
        },
        id="vaasa",
    ),
    pytest.param(
        "oulu",
        {
            "temperature_c": (1.4, 1.3),
            "wind_speed_ms": (9.1, 6.9),
            "radiation_wm2": (-1.0, -1.0),
        },
        id="oulu",
    ),
]


class StubClient:
    """Remote client that returns fixed observations and records calls."""

    def __init__(
        self,
        weather: list[tuple[datetime, dict[str, float | None]]] | None = None,
        radiation: list[tuple[datetime, float | None]] | None = None,
    ) -> None:
        self._weather = list(weather or [])
        self._radiation = list(radiation or [])
        self.weather_calls: list[tuple[int, datetime, datetime]] = []
        self.radiation_calls: list[tuple[int, datetime, datetime]] = []
        self.closed = False

    def iter_weather(
        self, fmisid: int, start: datetime, end: datetime
    ) -> list[tuple[datetime, dict[str, float | None]]]:
        self.weather_calls.append((fmisid, start, end))
        return list(self._weather)

    def iter_radiation(
        self, fmisid: int, start: datetime, end: datetime
    ) -> list[tuple[datetime, float | None]]:
        self.radiation_calls.append((fmisid, start, end))
        return list(self._radiation)

    def close(self) -> None:
        self.closed = True


def make_fetcher(
    city: str,
    *,
    local_path: str | Path | None | object = _UNSET,
    client: StubClient | None = None,
) -> FmiWeatherFetcher:
    options: dict[str, Any] = {"client": client or StubClient()}
    if local_path is not _UNSET:
        options["local_path"] = local_path
    return FmiWeatherFetcher(city, **options)


def write_local(
    path: Path,
    rows: list[tuple[datetime, dict[str, float | None]]],
    *,
    tz: str | None = "UTC",
) -> Path:
    """Writes a local snapshot with the same layout as a bundled parquet file."""
    schema = pa.schema([
        pa.field("timestamp_utc", pa.timestamp("us", tz=tz)),
        pa.field("temperature_c", pa.float32()),
        pa.field("wind_speed_ms", pa.float32()),
        pa.field("radiation_wm2", pa.float32()),
    ])
    table = pa.Table.from_pydict(
        {
            "timestamp_utc": [timestamp for timestamp, _ in rows],
            **{
                column: [values.get(column) for _, values in rows]
                for column in ("temperature_c", "wind_speed_ms", "radiation_wm2")
            },
        },
        schema=schema,
    )
    pq.write_table(table, path)
    return path


def test_fetcher_satisfies_datasource_protocol():
    assert isinstance(make_fetcher("helsinki"), DataSourceFetcher)


@pytest.mark.parametrize(("city", "expected_values"), CITY_CASES)
def test_source_name_and_schema(city, expected_values):
    fetcher = make_fetcher(city)

    assert fetcher.source_name == f"fmi_weather_{city}"
    assert fetcher.expected_schema.names == [
        "timestamp_utc",
        "temperature_c",
        "wind_speed_ms",
        "radiation_wm2",
    ]
    assert fetcher.expected_schema.field("timestamp_utc").type == pa.timestamp("us", tz="UTC")
    assert all(
        fetcher.expected_schema.field(column).type == pa.float32()
        for column in expected_values
    )


@pytest.mark.parametrize(("city", "expected_values"), CITY_CASES)
def test_bundled_sample_is_committed(city, expected_values):
    path = DEFAULT_LOCAL_PATHS[city]

    assert path.is_file()
    assert path.stat().st_size > 0


def test_city_aliases_and_unsupported_cities():
    assert station_for_city("Helsinki") is STATIONS["helsinki"]
    assert station_for_city("Kaisaniemi") is STATIONS["helsinki"]
    assert station_for_city("Helsinki (Kaisaniemi)") is STATIONS["helsinki"]
    assert station_for_city("Vaasa") is STATIONS["vaasa"]

    with pytest.raises(ValueError, match="Unsupported FMI city 'Turku'"):
        FmiWeatherFetcher("Turku", client=StubClient())


def test_vaasa_uses_a_multi_year_radiation_station():
    assert station_for_city("vaasa").radiation_fmisid == 101486
    assert station_for_city("vaasa").weather_fmisid == 101485


@pytest.mark.parametrize(("city", "expected_values"), CITY_CASES)
def test_fetch_from_local_resamples_bundled_day(city, expected_values):
    client = StubClient()
    fetcher = FmiWeatherFetcher(city, client=client)  # type: ignore[arg-type]

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

    for column, (first, last) in expected_values.items():
        values = table[column].to_pylist()
        assert values[0] == pytest.approx(first, abs=1e-5)
        assert values[-1] == pytest.approx(last, abs=1e-5)

    assert client.weather_calls == []
    assert client.radiation_calls == []


def test_fetch_merges_weather_and_radiation_by_timestamp():
    weather = [
        (
            START,
            {"TA_PT1H_AVG": 1.5, "WS_PT1H_AVG": 2.5},
        ),
        (START + timedelta(hours=1), {"TA_PT1H_AVG": 3.5}),
    ]
    radiation = [
        (START, 100.0),
        (START + timedelta(hours=1), None),
        (END, 999.0),
    ]
    client = StubClient(weather, radiation)
    fetcher = make_fetcher("helsinki", local_path=None, client=client)

    table = fetcher.fetch(FetchRequest(START, END))

    assert table.to_pylist() == [
        {
            "timestamp_utc": START,
            "temperature_c": 1.5,
            "wind_speed_ms": 2.5,
            "radiation_wm2": 100.0,
        },
        {
            "timestamp_utc": START + timedelta(hours=1),
            "temperature_c": 3.5,
            "wind_speed_ms": None,
            "radiation_wm2": None,
        },
    ]
    assert client.weather_calls == [(100971, START, END)]
    assert client.radiation_calls == [(101004, START, END)]


def test_fetch_keeps_weather_rows_when_radiation_is_missing():
    client = StubClient([(START, {"TA_PT1H_AVG": 2.0})], [])
    fetcher = make_fetcher("oulu", local_path=None, client=client)

    table = fetcher.fetch(FetchRequest(START, END))

    assert table.num_rows == 1
    assert table["temperature_c"].to_pylist() == [pytest.approx(2.0)]
    assert table["wind_speed_ms"].to_pylist() == [None]
    assert table["radiation_wm2"].to_pylist() == [None]


def test_fetch_normalizes_request_timezone_to_utc():
    client = StubClient([(START, {"TA_PT1H_AVG": 2.0})], [])
    fetcher = make_fetcher("helsinki", local_path=None, client=client)
    east_two = timezone(timedelta(hours=2))

    table = fetcher.fetch(
        FetchRequest(
            datetime(2024, 1, 1, 2, 0, tzinfo=east_two),
            datetime(2024, 1, 1, 4, 0, tzinfo=east_two),
        )
    )

    assert table["timestamp_utc"].to_pylist() == [START]
    assert client.weather_calls == [(100971, START, END)]


def test_fetch_from_local_falls_back_when_snapshot_is_missing(tmp_path):
    client = StubClient([(START, {"TA_PT1H_AVG": 2.0})], [(START, 10.0)])
    fetcher = make_fetcher(
        "helsinki",
        local_path=tmp_path / "missing.parquet",
        client=client,
    )

    table = fetcher.fetch_from_local(FetchRequest(START, END))

    assert table.num_rows == 1
    assert client.weather_calls == [(100971, START, END)]
    assert client.radiation_calls == [(101004, START, END)]


def test_fetch_from_local_falls_back_when_disabled():
    client = StubClient([(START, {"TA_PT1H_AVG": 2.0})], [(START, 10.0)])
    fetcher = make_fetcher("helsinki", local_path=None, client=client)

    table = fetcher.fetch_from_local(FetchRequest(START, END))

    assert table.num_rows == 1
    assert len(client.weather_calls) == 1


def test_fetch_from_local_averages_subhourly_values(tmp_path):
    rows = [
        (START, {"temperature_c": 1.0, "wind_speed_ms": 2.0, "radiation_wm2": 100.0}),
        (
            START + timedelta(minutes=30),
            {"temperature_c": 2.0, "wind_speed_ms": 4.0, "radiation_wm2": 200.0},
        ),
        (START + timedelta(hours=1), {"temperature_c": 3.0}),
    ]
    fetcher = make_fetcher(
        "helsinki",
        local_path=write_local(tmp_path / "weather.parquet", rows),
    )

    table = fetcher.fetch_from_local(FetchRequest(START, END))

    assert table["temperature_c"].to_pylist() == [
        pytest.approx(1.5),
        pytest.approx(3.0),
    ]
    assert table["wind_speed_ms"].to_pylist() == [pytest.approx(3.0), None]
    assert table["radiation_wm2"].to_pylist() == [pytest.approx(150.0), None]


def test_fetch_from_local_assumes_utc_for_naive_timestamps(tmp_path):
    rows = [(datetime(2024, 1, 1, 0, 0), {"temperature_c": 1.0})]
    fetcher = make_fetcher(
        "helsinki",
        local_path=write_local(tmp_path / "weather.parquet", rows, tz=None),
    )

    table = fetcher.fetch_from_local(FetchRequest(START, END))

    assert table["timestamp_utc"].to_pylist() == [START]
    assert table["temperature_c"].to_pylist() == [pytest.approx(1.0)]


def test_fetch_returns_empty_table_for_window_without_data(tmp_path):
    fetcher = make_fetcher(
        "helsinki",
        local_path=write_local(
            tmp_path / "weather.parquet",
            [(START, {"temperature_c": 1.0})],
        ),
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
    fetcher = make_fetcher("helsinki", local_path=None)

    with pytest.raises(ValueError, match=message):
        fetcher.fetch(request_)


def test_context_manager_closes_client():
    client = StubClient()
    fetcher = make_fetcher("helsinki", local_path=None, client=client)

    with fetcher:
        pass

    assert client.closed