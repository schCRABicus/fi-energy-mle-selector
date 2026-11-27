from __future__ import annotations

import math
import os
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta

import pytest
from dotenv import load_dotenv

from fi_energy.datasets import FetchRequest, FingridWindCapacityFetcher

load_dotenv()

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
def fetcher() -> Iterator[FingridWindCapacityFetcher]:
    fetcher = FingridWindCapacityFetcher(os.environ["FINGRID_API_KEY"])
    yield fetcher
    fetcher.close()


def test_live_hourly_capacity_fetch_returns_validated_dataset(fetcher):
    table = fetcher.fetch(FetchRequest(start_utc=WINDOW_START, end_utc=WINDOW_END))

    assert table.schema.equals(fetcher.expected_schema, check_metadata=False)
    assert table.num_rows == 24
    assert table.column("timestamp_utc").to_pylist() == [
        WINDOW_START + timedelta(hours=hour) for hour in range(24)
    ]

    values = table.column("wind_capacity_mw").to_pylist()
    assert all(math.isfinite(value) for value in values)
    assert all(value > 0 for value in values)