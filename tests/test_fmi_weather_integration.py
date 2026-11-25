from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import pytest

from datasets import FetchRequest, FmiWeatherFetcher
from datasets.fmi.weather import VALUE_COLUMNS

WINDOW_START = datetime(2024, 3, 1, tzinfo=UTC)
WINDOW_END = datetime(2024, 3, 2, tzinfo=UTC)

pytestmark = pytest.mark.integration

CITIES = [
    pytest.param("helsinki", id="helsinki"),
    pytest.param("vaasa", id="vaasa"),
    pytest.param("oulu", id="oulu"),
]


@pytest.mark.parametrize("city", CITIES)
def test_live_hourly_fetch_returns_validated_dataset(city):
    with FmiWeatherFetcher(city) as fetcher:
        table = fetcher.fetch(FetchRequest(start_utc=WINDOW_START, end_utc=WINDOW_END))

    assert fetcher.source_name == f"fmi_weather_{city}"
    assert table.schema.equals(fetcher.expected_schema, check_metadata=False)
    assert table.num_rows == 24
    assert table["timestamp_utc"].to_pylist() == [
        WINDOW_START + timedelta(hours=hour) for hour in range(24)
    ]

    for column, (low, high) in {
        "temperature_c": (-60.0, 40.0),
        "wind_speed_ms": (0.0, 50.0),
        "radiation_wm2": (-10.0, 1_500.0),
    }.items():
        values = [value for value in table[column].to_pylist() if value is not None]
        assert len(values) >= 20
        assert all(math.isfinite(value) and low <= value <= high for value in values)


@pytest.mark.parametrize("city", CITIES)
def test_live_hourly_fetch_matches_committed_snapshot(city):
    request = FetchRequest(start_utc=WINDOW_START, end_utc=WINDOW_END)
    with FmiWeatherFetcher(city) as fetcher:
        live = fetcher.fetch(request)
        local = fetcher.fetch_from_local(request)

    assert live["timestamp_utc"].to_pylist() == local["timestamp_utc"].to_pylist()
    for column in VALUE_COLUMNS:
        for live_value, local_value in zip(
            live[column].to_pylist(),
            local[column].to_pylist(),
            strict=True,
        ):
            if live_value is None or local_value is None:
                assert live_value is local_value
            else:
                assert live_value == pytest.approx(
                    local_value, rel=1e-5, abs=0.01
                )