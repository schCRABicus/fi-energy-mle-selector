from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from datasets import FetchRequest, NordPoolSpotFetcher

WINDOW_START = datetime(2024, 3, 1, tzinfo=UTC)
WINDOW_END = datetime(2024, 3, 2, tzinfo=UTC)

pytestmark = pytest.mark.integration


def test_live_one_day_fetch_matches_committed_snapshot():
    request = FetchRequest(start_utc=WINDOW_START, end_utc=WINDOW_END)
    with NordPoolSpotFetcher() as fetcher:
        live = fetcher.fetch(request)
        local = fetcher.fetch_from_local(request)

    assert live.schema.equals(fetcher.expected_schema, check_metadata=False)
    assert live.num_rows == 24
    assert live["timestamp_utc"].to_pylist() == local["timestamp_utc"].to_pylist()
    for live_value, local_value in zip(
        live["spot_price_c_kwh"].to_pylist(),
        local["spot_price_c_kwh"].to_pylist(),
        strict=True,
    ):
        if live_value is None or local_value is None:
            assert live_value is local_value
        else:
            assert live_value == pytest.approx(local_value, rel=1e-5, abs=1e-5)


def test_live_three_year_window_covers_every_hour():
    start = datetime(2022, 1, 1, tzinfo=UTC)
    end = datetime(2025, 1, 1, tzinfo=UTC)
    with NordPoolSpotFetcher() as fetcher:
        table = fetcher.fetch(FetchRequest(start_utc=start, end_utc=end))

    assert table.schema.equals(fetcher.expected_schema, check_metadata=False)
    assert table.num_rows == 26_304
    assert table["timestamp_utc"].to_pylist() == [
        start + timedelta(hours=hour) for hour in range(26_304)
    ]

    values = table["spot_price_c_kwh"].to_pylist()
    assert all(value is not None for value in values)
    assert all(-100.0 <= value <= 1_000.0 for value in values)