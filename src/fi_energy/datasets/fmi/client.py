from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from datetime import UTC, datetime, timedelta
from typing import Any
from xml.etree import ElementTree

import httpx
from tenacity import RetryCallState

from fi_energy.datasets.http_client import HttpClient
from fi_energy.datasets.progress import ChunkProgress
from fi_energy.datasets.utils import utc_window

FMI_API_BASE_URL = "https://opendata.fmi.fi/"
WFS_PATH = "wfs"

WEATHER_STORED_QUERY = "fmi::observations::weather::hourly::simple"
WEATHER_PARAMETERS = ("TA_PT1H_AVG", "WS_PT1H_AVG")

RADIATION_STORED_QUERY = "fmi::observations::radiation::simple"
RADIATION_PARAMETER = "GLOB_1MIN"
RADIATION_TIMESTEP_MINUTES = 60

# FMI caps the length of a single WFS request: 744 hours for hourly weather
# observations and 168 hours for one-minute radiation observations.
WEATHER_CHUNK_DAYS = 31
RADIATION_CHUNK_DAYS = 7

_NAMESPACES = {"BsWfs": "http://xml.fmi.fi/schema/wfs/2.0"}
_WEATHER_PARAMETER_SET = frozenset(WEATHER_PARAMETERS)


class FmiClient(HttpClient):
    """FMI Open Data WFS client for station observations."""

    def __init__(
        self,
        *,
        timeout: float = 60.0,
        max_retries: int = 5,
        transport: httpx.BaseTransport | None = None,
        retry_wait: Callable[[RetryCallState], float] | None = None,
    ) -> None:
        super().__init__(
            FMI_API_BASE_URL,
            headers={"User-Agent": "fi-energy-mle-selector/0.1"},
            timeout=timeout,
            max_retries=max_retries,
            transport=transport,
            retry_wait=retry_wait,
        )

    def iter_weather(
        self,
        fmisid: int,
        start: datetime,
        end: datetime,
        *,
        progress: ChunkProgress | None = None,
    ) -> Iterator[tuple[datetime, dict[str, float | None]]]:
        """
        Yield hourly weather observations as ``(timestamp_utc, values)``.

        ``values`` maps the FMI parameter names ``TA_PT1H_AVG`` and
        ``WS_PT1H_AVG`` to their readings. A parameter missing from an
        observation is simply absent from the mapping.

        ``progress`` is called before the first request and after every chunk
        with ``(stage, completed_chunks, total_chunks, observations)``.
        """
        start, end = utc_window(start, end)
        chunks = list(_iter_chunks(start, end, WEATHER_CHUNK_DAYS))
        observations = 0
        _report(progress, "weather", 0, len(chunks), observations)

        for completed, (chunk_start, chunk_end) in enumerate(chunks, start=1):
            rows: dict[datetime, dict[str, float | None]] = {}
            elements = self._get_elements(
                stored_query=WEATHER_STORED_QUERY,
                fmisid=fmisid,
                parameters=",".join(WEATHER_PARAMETERS),
                start=chunk_start,
                end=chunk_end,
            )
            for timestamp, parameter, value in elements:
                if parameter not in _WEATHER_PARAMETER_SET:
                    raise ValueError(f"Unexpected FMI weather parameter: {parameter!r}.")
                if chunk_start <= timestamp < chunk_end:
                    rows.setdefault(timestamp, {})[parameter] = value

            observations += len(rows)
            _report(progress, "weather", completed, len(chunks), observations)
            for timestamp in sorted(rows):
                yield timestamp, rows[timestamp]

    def iter_radiation(
        self,
        fmisid: int,
        start: datetime,
        end: datetime,
        *,
        progress: ChunkProgress | None = None,
    ) -> Iterator[tuple[datetime, float | None]]:
        """
        Yield hourly global-radiation samples as ``(timestamp_utc, W/m²)``.

        FMI's radiation archive stores one-minute samples, so ``timestep=60``
        returns the sample at the top of each hour rather than an hourly mean.

        ``progress`` behaves as in :meth:`iter_weather` with stage ``radiation``.
        """
        start, end = utc_window(start, end)
        chunks = list(_iter_chunks(start, end, RADIATION_CHUNK_DAYS))
        observations = 0
        _report(progress, "radiation", 0, len(chunks), observations)

        for completed, (chunk_start, chunk_end) in enumerate(chunks, start=1):
            values: dict[datetime, float | None] = {}
            elements = self._get_elements(
                stored_query=RADIATION_STORED_QUERY,
                fmisid=fmisid,
                parameters=RADIATION_PARAMETER,
                start=chunk_start,
                end=chunk_end,
                timestep=RADIATION_TIMESTEP_MINUTES,
            )
            for timestamp, parameter, value in elements:
                if parameter != RADIATION_PARAMETER:
                    raise ValueError(f"Unexpected FMI radiation parameter: {parameter!r}.")
                if chunk_start <= timestamp < chunk_end:
                    values[timestamp] = value

            observations += len(values)
            _report(progress, "radiation", completed, len(chunks), observations)
            for item in sorted(values.items()):
                yield item

    def _get_elements(
        self,
        *,
        stored_query: str,
        fmisid: int,
        parameters: str,
        start: datetime,
        end: datetime,
        timestep: int | None = None,
    ) -> Iterator[tuple[datetime, str, float | None]]:
        params: dict[str, Any] = {
            "service": "WFS",
            "version": "2.0.0",
            "request": "getFeature",
            "storedquery_id": stored_query,
            "fmisid": fmisid,
            "parameters": parameters,
            "starttime": _format_timestamp(start),
            "endtime": _format_timestamp(end),
        }
        if timestep is not None:
            params["timestep"] = timestep

        xml = self.get_text(WFS_PATH, params=params)
        yield from _parse_elements(xml)


def _report(
    progress: ChunkProgress | None,
    stage: str,
    completed: int,
    total: int,
    observations: int,
) -> None:
    if progress is not None:
        progress(stage, completed, total, observations)


def _iter_chunks(
    start: datetime, end: datetime, days: int
) -> Iterator[tuple[datetime, datetime]]:
    cursor = start
    while cursor < end:
        chunk_end = min(cursor + timedelta(days=days), end)
        yield cursor, chunk_end
        cursor = chunk_end


def _parse_elements(xml: str) -> Iterator[tuple[datetime, str, float | None]]:
    root = ElementTree.fromstring(xml)
    if _local_name(root.tag) == "ExceptionReport":
        message = " ".join(part.strip() for part in root.itertext() if part.strip())
        raise ValueError(f"FMI WFS request failed: {message}")

    for element in root.findall(".//BsWfs:BsWfsElement", _NAMESPACES):
        raw_timestamp = element.findtext("BsWfs:Time", namespaces=_NAMESPACES)
        parameter = element.findtext("BsWfs:ParameterName", namespaces=_NAMESPACES)
        if raw_timestamp is None:
            raise ValueError("FMI WFS element is missing BsWfs:Time.")
        if parameter is None:
            raise ValueError("FMI WFS element is missing BsWfs:ParameterName.")

        value = element.findtext("BsWfs:ParameterValue", namespaces=_NAMESPACES)
        yield _parse_timestamp(raw_timestamp), parameter.strip(), _parse_value(value)


def _parse_timestamp(raw_timestamp: str) -> datetime:
    timestamp = datetime.fromisoformat(raw_timestamp.strip())
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=UTC)
    return timestamp.astimezone(UTC)


def _parse_value(raw_value: str | None) -> float | None:
    if raw_value is None or not raw_value.strip():
        return None
    value = float(raw_value)
    return None if math.isnan(value) else value


def _format_timestamp(timestamp: datetime) -> str:
    return timestamp.astimezone(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]