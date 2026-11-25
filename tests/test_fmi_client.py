from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
import pytest
from tenacity import wait_none

from datasets import FmiClient

START = datetime(2024, 1, 1, 0, 0, tzinfo=UTC)
END = datetime(2024, 1, 1, 2, 0, tzinfo=UTC)

WFS_NS = "http://www.opengis.net/wfs/2.0"
BSWFS_NS = "http://xml.fmi.fi/schema/wfs/2.0"


def element(timestamp: str | None, parameter: str | None, value: str | None) -> str:
    fields = []
    if timestamp is not None:
        fields.append(f"<BsWfs:Time>{timestamp}</BsWfs:Time>")
    if parameter is not None:
        fields.append(f"<BsWfs:ParameterName>{parameter}</BsWfs:ParameterName>")
    fields.append(
        "<BsWfs:ParameterValue/>"
        if value is None
        else f"<BsWfs:ParameterValue>{value}</BsWfs:ParameterValue>"
    )
    return f"<wfs:member><BsWfs:BsWfsElement>{''.join(fields)}</BsWfs:BsWfsElement></wfs:member>"


def feature_collection(*elements: str) -> str:
    return (
        f'<wfs:FeatureCollection xmlns:wfs="{WFS_NS}" xmlns:BsWfs="{BSWFS_NS}">'
        f"{''.join(elements)}"
        "</wfs:FeatureCollection>"
    )


EMPTY_COLLECTION = feature_collection()


class FakeFmiApi:
    """Scripted FMI WFS served through httpx.MockTransport."""

    def __init__(self, responses: list[str] | None = None) -> None:
        self._responses = list(responses or [])
        self.requests: list[httpx.Request] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        text = self._responses.pop(0) if self._responses else EMPTY_COLLECTION
        return httpx.Response(200, text=text)

    @property
    def urls(self) -> list[httpx.URL]:
        return [request.url for request in self.requests]


def make_client(api: FakeFmiApi, **options: Any) -> FmiClient:
    return FmiClient(
        transport=httpx.MockTransport(api.handler),
        retry_wait=wait_none(),
        **options,
    )


def test_iter_weather_groups_parameters_and_normalizes_timestamps():
    api = FakeFmiApi([
        feature_collection(
            element("2024-01-01T02:00:00+02:00", "TA_PT1H_AVG", "1.5"),
            element("2024-01-01T02:00:00+02:00", "WS_PT1H_AVG", "2.5"),
            element("2024-01-01T01:00:00Z", "TA_PT1H_AVG", "NaN"),
            element("2024-01-01T01:00:00Z", "WS_PT1H_AVG", None),
        ),
    ])

    assert list(make_client(api).iter_weather(100971, START, END)) == [
        (
            datetime(2024, 1, 1, 0, 0, tzinfo=UTC),
            {"TA_PT1H_AVG": 1.5, "WS_PT1H_AVG": 2.5},
        ),
        (
            datetime(2024, 1, 1, 1, 0, tzinfo=UTC),
            {"TA_PT1H_AVG": None, "WS_PT1H_AVG": None},
        ),
    ]


def test_iter_radiation_maps_nan_to_none_and_keeps_negative_values():
    api = FakeFmiApi([
        feature_collection(
            element("2024-01-01T00:00:00Z", "GLOB_1MIN", "-0.5"),
            element("2024-01-01T01:00:00Z", "GLOB_1MIN", "NaN"),
            element("2024-01-01T02:00:00Z", "GLOB_1MIN", "3.2"),
        ),
    ])

    assert list(make_client(api).iter_radiation(101004, START, END)) == [
        (datetime(2024, 1, 1, 0, 0, tzinfo=UTC), -0.5),
        (datetime(2024, 1, 1, 1, 0, tzinfo=UTC), None),
    ]


def test_client_sends_weather_wfs_parameters():
    api = FakeFmiApi()
    client = make_client(api)

    list(client.iter_weather(100971, START, END))

    request = api.requests[0]
    assert request.url.path == "/wfs"
    assert request.url.params["service"] == "WFS"
    assert request.url.params["version"] == "2.0.0"
    assert request.url.params["request"] == "getFeature"
    assert request.url.params["storedquery_id"] == "fmi::observations::weather::hourly::simple"
    assert request.url.params["fmisid"] == "100971"
    assert request.url.params["parameters"] == "TA_PT1H_AVG,WS_PT1H_AVG"
    assert request.url.params["starttime"] == "2024-01-01T00:00:00Z"
    assert request.url.params["endtime"] == "2024-01-01T02:00:00Z"
    assert "timestep" not in request.url.params


def test_client_sends_radiation_wfs_parameters():
    api = FakeFmiApi()
    client = make_client(api)

    list(client.iter_radiation(101004, START, END))

    request = api.requests[0]
    assert request.url.params["storedquery_id"] == "fmi::observations::radiation::simple"
    assert request.url.params["fmisid"] == "101004"
    assert request.url.params["parameters"] == "GLOB_1MIN"
    assert request.url.params["timestep"] == "60"


def test_weather_requests_are_chunked_to_at_most_31_days():
    api = FakeFmiApi()

    list(
        make_client(api).iter_weather(
            100971,
            datetime(2024, 1, 1, tzinfo=UTC),
            datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=32),
        )
    )

    assert len(api.requests) == 2
    assert api.requests[0].url.params["starttime"] == "2024-01-01T00:00:00Z"
    assert api.requests[0].url.params["endtime"] == "2024-02-01T00:00:00Z"
    assert api.requests[1].url.params["starttime"] == "2024-02-01T00:00:00Z"
    assert api.requests[1].url.params["endtime"] == "2024-02-02T00:00:00Z"


def test_radiation_requests_are_chunked_to_at_most_7_days():
    api = FakeFmiApi()
    start = datetime(2024, 1, 1, tzinfo=UTC)

    list(make_client(api).iter_radiation(101004, start, start + timedelta(days=8)))

    assert len(api.requests) == 2
    assert api.requests[0].url.params["starttime"] == "2024-01-01T00:00:00Z"
    assert api.requests[0].url.params["endtime"] == "2024-01-08T00:00:00Z"
    assert api.requests[1].url.params["starttime"] == "2024-01-08T00:00:00Z"
    assert api.requests[1].url.params["endtime"] == "2024-01-09T00:00:00Z"


def test_iter_weather_reports_chunk_progress():
    api = FakeFmiApi([EMPTY_COLLECTION, EMPTY_COLLECTION])
    events: list[tuple[str, int, int, int]] = []
    start = datetime(2024, 1, 1, tzinfo=UTC)

    list(
        make_client(api).iter_weather(
            100971,
            start,
            start + timedelta(days=32),
            progress=lambda *event: events.append(event),
        )
    )

    assert events == [
        ("weather", 0, 2, 0),
        ("weather", 1, 2, 0),
        ("weather", 2, 2, 0),
    ]


def test_iter_radiation_reports_chunk_progress_with_observation_count():
    api = FakeFmiApi([
        feature_collection(element("2024-01-01T00:00:00Z", "GLOB_1MIN", "1.0")),
        feature_collection(element("2024-01-08T00:00:00Z", "GLOB_1MIN", "2.0")),
    ])
    events: list[tuple[str, int, int, int]] = []
    start = datetime(2024, 1, 1, tzinfo=UTC)

    list(
        make_client(api).iter_radiation(
            101004,
            start,
            start + timedelta(days=8),
            progress=lambda *event: events.append(event),
        )
    )

    assert events == [
        ("radiation", 0, 2, 0),
        ("radiation", 1, 2, 1),
        ("radiation", 2, 2, 2),
    ]


def test_chunk_boundaries_do_not_duplicate_observations():
    api = FakeFmiApi([
        feature_collection(element("2024-01-08T00:00:00Z", "GLOB_1MIN", "1.0")),
        feature_collection(element("2024-01-08T00:00:00Z", "GLOB_1MIN", "2.0")),
    ])
    start = datetime(2024, 1, 1, tzinfo=UTC)

    samples = list(make_client(api).iter_radiation(101004, start, start + timedelta(days=8)))

    assert samples == [(datetime(2024, 1, 8, tzinfo=UTC), 2.0)]


def test_client_raises_on_wfs_exception_report():
    body = (
        '<ows:ExceptionReport xmlns:ows="http://www.opengis.net/ows/1.1">'
        "<ows:Exception><ows:ExceptionText>Too long time interval requested!</ows:ExceptionText>"
        "</ows:Exception></ows:ExceptionReport>"
    )
    api = FakeFmiApi([body])

    with pytest.raises(ValueError, match="Too long time interval requested"):
        list(make_client(api).iter_weather(100971, START, END))


def test_client_rejects_unexpected_weather_parameter():
    api = FakeFmiApi([feature_collection(element("2024-01-01T00:00:00Z", "BAD", "1.0"))])

    with pytest.raises(ValueError, match="Unexpected FMI weather parameter"):
        list(make_client(api).iter_weather(100971, START, END))


def test_client_rejects_unexpected_radiation_parameter():
    api = FakeFmiApi([
        feature_collection(element("2024-01-01T00:00:00Z", "TA_PT1H_AVG", "1.0")),
    ])

    with pytest.raises(ValueError, match="Unexpected FMI radiation parameter"):
        list(make_client(api).iter_radiation(101004, START, END))


def test_client_rejects_element_without_timestamp():
    api = FakeFmiApi([feature_collection(element(None, "TA_PT1H_AVG", "1.0"))])

    with pytest.raises(ValueError, match="missing BsWfs:Time"):
        list(make_client(api).iter_weather(100971, START, END))


def test_client_rejects_element_without_parameter_name():
    api = FakeFmiApi([feature_collection(element("2024-01-01T00:00:00Z", None, "1.0"))])

    with pytest.raises(ValueError, match="missing BsWfs:ParameterName"):
        list(make_client(api).iter_weather(100971, START, END))