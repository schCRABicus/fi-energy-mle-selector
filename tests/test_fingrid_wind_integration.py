from __future__ import annotations

import math
import os
from collections import defaultdict
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta

import pytest
from dotenv import load_dotenv

from fi_energy.datasets import FetchRequest
from fi_energy.datasets.fingrid import FingridWindFetcher

load_dotenv()

# A closed window in the past: Fingrid serves consolidated data for it,
# and it is after Fingrid switched dataset 75 to a 15-minute resolution,
# so the hourly path exercises real resampling.
WINDOW_START = datetime(2024, 3, 1, tzinfo=UTC)
WINDOW_END = datetime(2024, 3, 2, tzinfo=UTC)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not os.getenv("FINGRID_API_KEY"),
        reason="FINGRID_API_KEY is not set (add it to .env)",
    ),
]


@pytest.fixture(scope="module")
def fetcher() -> Iterator[FingridWindFetcher]:
    fetcher = FingridWindFetcher(os.environ["FINGRID_API_KEY"])
    yield fetcher
    fetcher.close()


def test_live_hourly_fetch_returns_validated_dataset(fetcher):
    table = fetcher.fetch(FetchRequest(start_utc=WINDOW_START, end_utc=WINDOW_END))

    assert table.schema.equals(fetcher.expected_schema, check_metadata=False)
    assert table.num_rows == 24
    assert table.column("timestamp_utc").to_pylist() == [
        WINDOW_START + timedelta(hours=hour) for hour in range(24)
    ]

    values = table.column("wind_mw").to_pylist()
    assert all(math.isfinite(value) for value in values)
    assert all(value >= 0 for value in values)
    assert any(value > 0 for value in values)


def test_live_hourly_values_are_means_of_quarter_hourly_samples(fetcher):
    start = datetime(2024, 3, 1, 6, tzinfo=UTC)
    end = datetime(2024, 3, 1, 10, tzinfo=UTC)

    hourly = fetcher.fetch(FetchRequest(start_utc=start, end_utc=end))
    quarter_hourly = fetcher.fetch(
        FetchRequest(start_utc=start, end_utc=end, resolution="15min")
    )

    samples_by_hour: dict[datetime, list[float]] = defaultdict(list)
    for timestamp, value in zip(
        quarter_hourly.column("timestamp_utc").to_pylist(),
        quarter_hourly.column("wind_mw").to_pylist(),
        strict=True,
    ):
        samples_by_hour[timestamp.replace(minute=0)].append(value)

    assert len(samples_by_hour) == 4
    assert all(len(samples) == 4 for samples in samples_by_hour.values())

    for timestamp, value in zip(
        hourly.column("timestamp_utc").to_pylist(),
        hourly.column("wind_mw").to_pylist(),
        strict=True,
    ):
        samples = samples_by_hour[timestamp]
        assert value == pytest.approx(sum(samples) / len(samples), rel=1e-5)