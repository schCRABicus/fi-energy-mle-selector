from dataclasses import FrozenInstanceError
from datetime import UTC, datetime

import pytest

import datasets
from datasets import DataSourceFetcher, FetchRequest


def test_fetch_request_defaults_to_hourly_resolution():
    request = FetchRequest(
        start_utc=datetime(2024, 1, 1, tzinfo=UTC),
        end_utc=datetime(2024, 2, 1, tzinfo=UTC),
    )

    assert request.resolution == "1h"


def test_fetch_request_is_immutable():
    request = FetchRequest(
        start_utc=datetime(2024, 1, 1, tzinfo=UTC),
        end_utc=datetime(2024, 2, 1, tzinfo=UTC),
    )

    with pytest.raises(FrozenInstanceError):
        request.resolution = "15min"  # type: ignore[misc]


def test_public_package_exports_the_api_and_fetchers():
    assert datasets.FetchRequest is FetchRequest
    assert datasets.DataSourceFetcher is DataSourceFetcher
    assert set(datasets.__all__) == {
        "DataSourceFetcher",
        "FetchRequest",
        "FingridClient",
        "FingridConsumptionFetcher",
        "FingridEEFlowFetcher",
        "FingridHydroFetcher",
        "FingridNuclearFetcher",
        "FingridSE1FlowFetcher",
        "FingridSE3FlowFetcher",
        "FingridWindCapacityFetcher",
        "FingridWindFetcher",
        "FmiClient",
        "FmiWeatherFetcher",
        "EnergyChartsClient",
        "NordPoolSpotFetcher",
    }
