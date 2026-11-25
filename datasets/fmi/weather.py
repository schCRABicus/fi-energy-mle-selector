"""Hourly weather observations for the Finnish cities used by the model."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Self

import pyarrow as pa

from datasets.api import FetchRequest
from datasets.fmi.client import FmiClient
from datasets.progress import ChunkProgress, DownloadTracker
from datasets.utils import iter_local_samples, mean_by_bucket, resolution_step, utc_window

DATA_DIR = Path(__file__).parent / "data"

WEATHER_SCHEMA = pa.schema([
    pa.field("timestamp_utc", pa.timestamp("us", tz="UTC"), nullable=False),
    pa.field("temperature_c", pa.float32()),
    pa.field("wind_speed_ms", pa.float32()),
    pa.field("radiation_wm2", pa.float32()),
])

VALUE_COLUMNS = ("temperature_c", "wind_speed_ms", "radiation_wm2")

_WEATHER_COLUMNS = {
    "TA_PT1H_AVG": "temperature_c",
    "WS_PT1H_AVG": "wind_speed_ms",
}


@dataclass(frozen=True)
class FmiStation:
    """Weather and radiation stations used for one target city."""

    key: str
    city: str
    weather_name: str
    weather_fmisid: int
    radiation_name: str
    radiation_fmisid: int


STATIONS: dict[str, FmiStation] = {
    "helsinki": FmiStation(
        key="helsinki",
        city="Helsinki",
        weather_name="Helsinki Kaisaniemi",
        weather_fmisid=100971,
        radiation_name="Helsinki Kumpula",
        radiation_fmisid=101004,
    ),
    "vaasa": FmiStation(
        key="vaasa",
        city="Vaasa",
        weather_name="Vaasa Klemettilä",
        weather_fmisid=101485,
        radiation_name="Seinäjoki Pelmaa",
        radiation_fmisid=101486,
    ),
    "oulu": FmiStation(
        key="oulu",
        city="Oulu",
        weather_name="Oulu Vihreäsaari",
        weather_fmisid=101794,
        radiation_name="Oulu Vihreäsaari",
        radiation_fmisid=101794,
    ),
}

_CITY_ALIASES = {
    "helsinki (kaisaniemi)": "helsinki",
    "helsinki-kaisaniemi": "helsinki",
    "kaisaniemi": "helsinki",
}

DEFAULT_LOCAL_PATHS = {
    key: DATA_DIR / f"{key}_sample.parquet" for key in STATIONS
}

_DEFAULT_LOCAL_PATH = object()


def station_for_city(city: str) -> FmiStation:
    """Resolves a case-insensitive city name (or alias) to its station pair."""
    key = " ".join(city.strip().lower().split())
    key = _CITY_ALIASES.get(key, key)
    try:
        return STATIONS[key]
    except KeyError:
        supported = ", ".join(sorted(STATIONS))
        raise ValueError(
            f"Unsupported FMI city {city!r}; expected one of: {supported}."
        ) from None


class FmiWeatherFetcher:
    """
    Fetches hourly temperature, wind speed, and radiation for one Finnish city.

    Weather comes from the city's primary observation station. Radiation comes
    from the nearest station with a multi-year record, which can differ from the
    weather station (Helsinki Kumpula for Kaisaniemi, Seinäjoki Pelmaa for Vaasa).
    Radiation values are FMI's one-minute global-radiation sample taken at the
    top of each hour. Small negative values occur at night because of sensor
    offset and are kept as published.
    """

    RESOLUTIONS: dict[str, timedelta] = {"1h": timedelta(hours=1)}

    def __init__(
        self,
        city: str,
        *,
        client: FmiClient | None = None,
        local_path: str | Path | None | object = _DEFAULT_LOCAL_PATH,
        timeout: float = 60.0,
        max_retries: int = 5,
        show_progress: bool | None = None,
    ) -> None:
        self._station = station_for_city(city)
        if local_path is _DEFAULT_LOCAL_PATH:
            local_path = DEFAULT_LOCAL_PATHS[self._station.key]
        self._local_path = None if local_path is None else Path(local_path)
        self._show_progress = show_progress
        self._client = client or FmiClient(timeout=timeout, max_retries=max_retries)

    @property
    def source_name(self) -> str:
        return f"fmi_weather_{self._station.key}"

    @property
    def expected_schema(self) -> pa.Schema:
        return WEATHER_SCHEMA

    def fetch(self, request: FetchRequest) -> pa.Table:
        start, end = utc_window(request.start_utc, request.end_utc)
        resolution_step(self.RESOLUTIONS, request.resolution)
        with self._progress_tracker() as tracker:
            rows = self._remote_rows(start, end, progress=tracker.stage)
        return self._build_table(rows)

    def fetch_from_local(self, request: FetchRequest) -> pa.Table:
        start, end = utc_window(request.start_utc, request.end_utc)
        step = resolution_step(self.RESOLUTIONS, request.resolution)

        local_path = self._local_path
        if local_path is None or not local_path.exists():
            with self._progress_tracker() as tracker:
                rows = self._remote_rows(start, end, progress=tracker.stage)
            return self._build_table(rows)

        rows: dict[datetime, dict[str, float]] = {}
        for column in VALUE_COLUMNS:
            samples = mean_by_bucket(
                iter_local_samples(local_path, value_column=column),
                start=start,
                end=end,
                step=step,
            )
            for timestamp, value in samples:
                rows.setdefault(timestamp, {})[column] = value
        return self._build_table(_table_rows(rows))

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def _remote_rows(
        self,
        start: datetime,
        end: datetime,
        *,
        progress: ChunkProgress | None = None,
    ) -> list[dict[str, object]]:
        rows: dict[datetime, dict[str, float | None]] = {}

        for timestamp, values in self._client.iter_weather(
            self._station.weather_fmisid, start, end, progress=progress
        ):
            if start <= timestamp < end:
                row = rows.setdefault(timestamp, {})
                for parameter, column in _WEATHER_COLUMNS.items():
                    row[column] = values.get(parameter)

        for timestamp, value in self._client.iter_radiation(
            self._station.radiation_fmisid, start, end, progress=progress
        ):
            if start <= timestamp < end:
                rows.setdefault(timestamp, {})["radiation_wm2"] = value

        return _table_rows(rows)

    def _progress_tracker(self) -> DownloadTracker:
        station = self._station
        title = (
            f"{station.city}: weather {station.weather_name} ({station.weather_fmisid}), "
            f"radiation {station.radiation_name} ({station.radiation_fmisid})"
        )
        return DownloadTracker(title, enabled=self._show_progress)

    def _build_table(self, rows: list[dict[str, object]]) -> pa.Table:
        return pa.Table.from_pydict(
            {
                "timestamp_utc": [row["timestamp_utc"] for row in rows],
                **{
                    column: [row.get(column) for row in rows]
                    for column in VALUE_COLUMNS
                },
            },
            schema=WEATHER_SCHEMA,
        )


def _table_rows(
    rows: dict[datetime, dict[str, float | None]],
) -> list[dict[str, object]]:
    return [
        {"timestamp_utc": timestamp, **{column: values.get(column) for column in VALUE_COLUMNS}}
        for timestamp, values in sorted(rows.items())
    ]