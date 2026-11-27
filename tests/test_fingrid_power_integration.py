from __future__ import annotations

import math
import os
from datetime import UTC, datetime, timedelta

import pytest
from dotenv import load_dotenv

from fi_energy.datasets import (
    FetchRequest,
    FingridConsumptionFetcher,
    FingridHydroFetcher,
    FingridNuclearFetcher,
)

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

CASES = [
    pytest.param(FingridConsumptionFetcher, 124, "consumption_mw", id="consumption-124"),
    pytest.param(FingridNuclearFetcher, 188, "nuclear_mw", id="nuclear-188"),
    pytest.param(FingridHydroFetcher, 191, "hydro_mw", id="hydro-191"),
]


@pytest.mark.parametrize(("fetcher_class", "dataset_id", "value_column"), CASES)
def test_live_hourly_fetch_returns_validated_dataset(
    fetcher_class, dataset_id, value_column
):
    with fetcher_class(os.environ["FINGRID_API_KEY"]) as fetcher:
        table = fetcher.fetch(FetchRequest(start_utc=WINDOW_START, end_utc=WINDOW_END))

    assert fetcher.source_name == f"fingrid_{dataset_id}"
    assert table.schema.equals(fetcher.expected_schema, check_metadata=False)
    assert table.num_rows == 24
    assert table.column("timestamp_utc").to_pylist() == [
        WINDOW_START + timedelta(hours=hour) for hour in range(24)
    ]

    values = table.column(value_column).to_pylist()
    assert all(math.isfinite(value) for value in values)
    assert all(value > 0 for value in values)


@pytest.mark.parametrize(("fetcher_class", "dataset_id", "value_column"), CASES)
def test_live_hourly_fetch_matches_committed_snapshot(
    fetcher_class, dataset_id, value_column
):
    with fetcher_class(os.environ["FINGRID_API_KEY"]) as fetcher:
        live = fetcher.fetch(FetchRequest(start_utc=WINDOW_START, end_utc=WINDOW_END))
        local = fetcher.fetch_from_local(
            FetchRequest(start_utc=WINDOW_START, end_utc=WINDOW_END)
        )

    assert live.column(value_column).to_pylist() == pytest.approx(
        local.column(value_column).to_pylist(), rel=1e-5, abs=0.01
    )