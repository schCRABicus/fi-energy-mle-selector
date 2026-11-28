"""Standard Finnish hourly household load profiles for contract simulation."""

from __future__ import annotations

import calendar
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import pyarrow as pa
import pyarrow.compute as pc

from fi_energy.datasets.alignment import TIMESTAMP_COLUMN

__all__ = [
    "APARTMENT_2500",
    "BASE_TEMPERATURE_C",
    "DEFAULT_PROFILES",
    "DETACHED_HOUSE_18000",
    "HOURLY_KWH",
    "LOCAL_TIMEZONE",
    "TEMPERATURE_SMOOTHING_HOURS",
    "LoadProfileBuild",
    "LoadProfileSpec",
    "build_load_profile_table",
    "profile_annual_totals",
    "profile_table",
]

HOURLY_KWH = "hourly_kwh"
LOCAL_TIMEZONE = "Europe/Helsinki"
BASE_TEMPERATURE_C = 17.0
TEMPERATURE_SMOOTHING_HOURS = 6

# The daily shapes follow Finnish household routines in local wall-clock time:
# a modest morning rise, a pronounced evening peak, and a low overnight base.
# Weekends start later and stay flatter through the day.
_WEEKDAY_HOUSEHOLD_SHAPE: tuple[float, ...] = (
    0.55, 0.48, 0.43, 0.41, 0.43, 0.52,
    0.70, 1.00, 1.08, 0.95, 0.85, 0.85,
    0.90, 0.88, 0.86, 0.92, 1.06, 1.25,
    1.35, 1.30, 1.16, 1.00, 0.85, 0.70,
)
_WEEKEND_HOUSEHOLD_SHAPE: tuple[float, ...] = (
    0.55, 0.48, 0.44, 0.42, 0.42, 0.44,
    0.50, 0.58, 0.70, 0.85, 0.95, 1.00,
    1.00, 0.95, 0.92, 0.95, 1.05, 1.18,
    1.28, 1.30, 1.20, 1.05, 0.88, 0.70,
)
_HOT_WATER_SHAPE: tuple[float, ...] = (
    0.55, 0.45, 0.40, 0.40, 0.45, 0.60,
    0.95, 1.55, 1.80, 1.35, 1.00, 0.90,
    0.95, 0.90, 0.85, 0.90, 1.00, 1.25,
    1.55, 1.75, 1.60, 1.30, 1.00, 0.75,
)
_HOUSEHOLD_SEASONAL_AMPLITUDE = 0.12
_HOT_WATER_SEASONAL_AMPLITUDE = 0.06
_SEASONAL_PEAK_DAY = 10
_ANNUAL_TOLERANCE_KWH = 0.1
_KEY_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
_FLOAT32 = pa.float32()

_HOURS_PER_DAY = 24
_HOURS_PER_YEAR = 365 * _HOURS_PER_DAY
_HOURS_PER_LEAP_YEAR = 366 * _HOURS_PER_DAY


@dataclass(frozen=True, slots=True)
class LoadProfileSpec:
    """
    One standard household profile and its annual consumption split.

    ``household_kwh`` covers lighting, appliances, cooking, and electronics;
    ``hot_water_kwh`` covers electric domestic hot water; ``space_heating_kwh``
    covers electric space heating. The three components must sum to
    ``annual_kwh``. A profile without electric heating or electric hot water
    leaves those components at zero.
    """

    key: str
    label: str
    annual_kwh: float
    household_kwh: float
    hot_water_kwh: float
    space_heating_kwh: float
    description: str

    def __post_init__(self) -> None:
        if not isinstance(self.key, str) or not _KEY_PATTERN.match(self.key):
            raise ValueError(
                "profile.key must be a lowercase identifier such as "
                f"'apartment_2500', got {self.key!r}."
            )
        if not isinstance(self.label, str) or not self.label.strip():
            raise ValueError("profile.label must be a non-empty string.")
        if not isinstance(self.description, str) or not self.description.strip():
            raise ValueError("profile.description must be a non-empty string.")
        for name in (
            "annual_kwh",
            "household_kwh",
            "hot_water_kwh",
            "space_heating_kwh",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(
                    f"profile.{name} must be a finite non-negative number, "
                    f"got {value!r}."
                )
            object.__setattr__(self, name, value)
        if self.annual_kwh <= 0.0:
            raise ValueError("profile.annual_kwh must be positive.")
        components = self.household_kwh + self.hot_water_kwh + self.space_heating_kwh
        if not math.isclose(
            components, self.annual_kwh, rel_tol=0.0, abs_tol=1e-6
        ):
            raise ValueError(
                f"profile components ({components:,.3f} kWh) must sum to "
                f"annual_kwh ({self.annual_kwh:,.3f} kWh)."
            )

    @property
    def column(self) -> str:
        """The hourly-consumption column name for this profile."""
        return f"{self.key}_{HOURLY_KWH}"

    @property
    def shares(self) -> dict[str, float]:
        return {
            "household": self.household_kwh / self.annual_kwh,
            "hot_water": self.hot_water_kwh / self.annual_kwh,
            "space_heating": self.space_heating_kwh / self.annual_kwh,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "column": self.column,
            "annual_kwh": self.annual_kwh,
            "components_kwh": {
                "household": self.household_kwh,
                "hot_water": self.hot_water_kwh,
                "space_heating": self.space_heating_kwh,
            },
            "shares": self.shares,
            "description": self.description,
        }


APARTMENT_2500 = LoadProfileSpec(
    key="apartment_2500",
    label="Apartment, 2,500 kWh/year, no electric heating",
    annual_kwh=2_500.0,
    household_kwh=2_500.0,
    hot_water_kwh=0.0,
    space_heating_kwh=0.0,
    description=(
        "A small apartment where space heating and domestic hot water are "
        "supplied by the building, so the metered electricity is household "
        "consumption only: lighting, appliances, cooking, and electronics."
    ),
)

DETACHED_HOUSE_18000 = LoadProfileSpec(
    key="detached_house_18000",
    label="Electrically heated detached house, 18,000 kWh/year",
    annual_kwh=18_000.0,
    household_kwh=5_000.0,
    hot_water_kwh=3_000.0,
    space_heating_kwh=10_000.0,
    description=(
        "An electrically heated detached house: 5,000 kWh of household "
        "electricity, 3,000 kWh of electric domestic hot water, and 10,000 kWh "
        "of electric space heating that follows the outdoor temperature."
    ),
)

DEFAULT_PROFILES: tuple[LoadProfileSpec, ...] = (
    APARTMENT_2500,
    DETACHED_HOUSE_18000,
)

_KEYS = [profile.key for profile in DEFAULT_PROFILES]
if len(_KEYS) != len(set(_KEYS)):
    raise RuntimeError("Default load profile keys must be unique.")


@dataclass(frozen=True, slots=True)
class LoadProfileBuild:
    """The per-profile hourly kWh table plus the checks that passed."""

    table: pa.Table
    profiles: tuple[LoadProfileSpec, ...]
    temperature_column: str
    checks: dict[str, bool]


def build_load_profile_table(
    source: pa.Table,
    profiles: tuple[LoadProfileSpec, ...] = DEFAULT_PROFILES,
    *,
    temperature_column: str = "helsinki_temperature_c",
) -> LoadProfileBuild:
    """
    Builds ``timestamp_utc`` + one ``<key>_hourly_kwh`` column per profile.

    Each component follows a local-time daily shape and is scaled so that a
    fully covered calendar year totals its target energy, even when interior
    hours are missing. A partial year at the window edge is scaled in
    proportion to the hours it contributes. Space heating is
    proportional to ``max(0, 17 °C - T)`` after a six-hour trailing mean of the
    outdoor temperature, so it never uses future weather. The result contains
    no nulls, NaNs, infinities, or negative values, and the returned checks
    confirm complete profiles and exact annual totals.

    Raises:
        TypeError: If ``source`` is not a pyarrow Table or the temperature
            column is not floating.
        ValueError: If required columns are missing, timestamps are not
            strictly ascending whole UTC hours, the temperature has nulls or
            non-finite values, or a profile's raw component is all zero.
        RuntimeError: If the assembled table violates the profile contract.
    """
    profiles = _normalize_profiles(profiles)
    _require_source(source, temperature_column)
    timestamps = _timestamps(source)
    local = [
        timestamp.astimezone(ZoneInfo(LOCAL_TIMEZONE))
        for timestamp in timestamps
    ]
    temperatures = source[temperature_column].to_pylist()
    # Annual calibration follows the UTC year, the same boundary the dataset
    # and the chronological train/test split use. Local time only shapes the
    # within-day profile.
    years = [timestamp.year for timestamp in timestamps]
    year_hours: dict[int, int] = defaultdict(int)
    for year in years:
        year_hours[year] += 1
    covered = {
        year: _year_is_covered(timestamps, year) for year in year_hours
    }

    household_raw = [_household_shape(moment) for moment in local]
    hot_water_raw = [_hot_water_shape(moment) for moment in local]
    smoothed = _smoothed_temperatures(timestamps, temperatures)
    heating_raw = [
        max(0.0, BASE_TEMPERATURE_C - value) for value in smoothed
    ]

    columns: dict[str, pa.Array] = {}
    for profile in profiles:
        components: list[list[float]] = []
        if profile.household_kwh:
            components.append(
                _scale_component(
                    household_raw,
                    years,
                    year_hours,
                    covered,
                    profile.household_kwh,
                )
            )
        if profile.hot_water_kwh:
            components.append(
                _scale_component(
                    hot_water_raw,
                    years,
                    year_hours,
                    covered,
                    profile.hot_water_kwh,
                )
            )
        if profile.space_heating_kwh:
            components.append(
                _scale_component(
                    heating_raw,
                    years,
                    year_hours,
                    covered,
                    profile.space_heating_kwh,
                )
            )
        values = [
            sum(parts) for parts in zip(*components, strict=True)
        ]
        columns[profile.column] = pa.array(values, type=_FLOAT32)

    table = pa.Table.from_arrays(
        [source[TIMESTAMP_COLUMN], *columns.values()],
        names=[TIMESTAMP_COLUMN, *columns],
    )
    checks = {
        "timestamps_preserved": (
            table[TIMESTAMP_COLUMN].to_pylist() == timestamps
        ),
        "profiles_complete": _profiles_complete(table, profiles),
        "profiles_non_negative": _profiles_non_negative(table, profiles),
        "annual_targets_met": _annual_targets_met(
            table, profiles, years, year_hours, covered
        ),
    }
    if not all(checks.values()):
        failed = ", ".join(name for name, ok in checks.items() if not ok)
        raise RuntimeError(f"Load profile check(s) failed: {failed}.")

    return LoadProfileBuild(
        table=table,
        profiles=profiles,
        temperature_column=temperature_column,
        checks=checks,
    )


def profile_annual_totals(build: LoadProfileBuild) -> dict[str, dict[int, float]]:
    """Returns the realised annual energy per profile and calendar year."""
    years = [
        timestamp.year
        for timestamp in build.table[TIMESTAMP_COLUMN].to_pylist()
    ]
    totals: dict[str, dict[int, float]] = {}
    for profile in build.profiles:
        sums: dict[int, float] = defaultdict(float)
        for year, value in zip(
            years, build.table[profile.column].to_pylist(), strict=True
        ):
            sums[year] += value
        totals[profile.key] = dict(sorted(sums.items()))
    return totals


def profile_table(build: LoadProfileBuild, key: str) -> pa.Table:
    """Extracts one profile as ``timestamp_utc`` + ``hourly_kwh``."""
    profile = next(
        (item for item in build.profiles if item.key == key), None
    )
    if profile is None:
        available = ", ".join(item.key for item in build.profiles)
        raise KeyError(f"Unknown load profile {key!r}; available: {available}.")
    return pa.Table.from_arrays(
        [build.table[TIMESTAMP_COLUMN], build.table[profile.column]],
        names=[TIMESTAMP_COLUMN, HOURLY_KWH],
    )


def _normalize_profiles(
    profiles: tuple[LoadProfileSpec, ...] | list[LoadProfileSpec],
) -> tuple[LoadProfileSpec, ...]:
    if not profiles:
        raise ValueError("At least one load profile is required.")
    normalized = tuple(profiles)
    keys = [profile.key for profile in normalized]
    if len(keys) != len(set(keys)):
        raise ValueError("Load profile keys must be unique.")
    return normalized


def _require_source(source: pa.Table, temperature_column: str) -> None:
    if not isinstance(source, pa.Table):
        raise TypeError(
            f"Expected source to be a pyarrow Table, got "
            f"{type(source).__name__}."
        )
    if source.num_rows == 0:
        raise ValueError("Source table is empty.")
    required = {TIMESTAMP_COLUMN, temperature_column}
    missing = sorted(required - set(source.column_names))
    if missing:
        raise ValueError(
            "Source is missing required column(s): " + ", ".join(missing)
        )
    if not pa.types.is_floating(source.schema.field(temperature_column).type):
        raise TypeError(
            f"{temperature_column!r} must be floating, got "
            f"{source.schema.field(temperature_column).type}."
        )
    timestamps = _timestamps(source)
    for timestamp in timestamps:
        if timestamp.tzinfo is None:
            raise ValueError("Timestamps must be timezone-aware UTC values.")
        if (
            timestamp.minute
            or timestamp.second
            or timestamp.microsecond
        ):
            raise ValueError(
                f"Timestamp {timestamp.isoformat()} is not a whole hour."
            )
    if any(
        later <= earlier
        for earlier, later in zip(timestamps, timestamps[1:])
    ):
        raise ValueError("Timestamps must be unique and strictly ascending.")
    temperatures = source[temperature_column].to_pylist()
    nulls = sum(value is None for value in temperatures)
    if nulls:
        raise ValueError(
            f"{temperature_column!r} contains {nulls:,} null values; build "
            "profiles from the complete hourly dataset."
        )
    non_finite = sum(not math.isfinite(value) for value in temperatures)
    if non_finite:
        raise ValueError(
            f"{temperature_column!r} contains {non_finite:,} non-finite "
            "values."
        )


def _timestamps(source: pa.Table) -> list[datetime]:
    timestamps = source[TIMESTAMP_COLUMN].to_pylist()
    if any(timestamp is None for timestamp in timestamps):
        raise ValueError("Timestamps must not contain null values.")
    return timestamps


def _hours_in_year(year: int) -> int:
    return (
        _HOURS_PER_LEAP_YEAR
        if calendar.isleap(year)
        else _HOURS_PER_YEAR
    )


def _year_is_covered(timestamps: list[datetime], year: int) -> bool:
    """True when the window spans the whole UTC year, gaps included."""
    return (
        timestamps[0] <= datetime(year, 1, 1, tzinfo=UTC)
        and timestamps[-1] >= datetime(year, 12, 31, 23, tzinfo=UTC)
    )


def _target_kwh(
    annual_kwh: float,
    year: int,
    year_hours: dict[int, int],
    covered: dict[int, bool],
) -> float:
    if covered[year]:
        return annual_kwh
    return annual_kwh * year_hours[year] / _hours_in_year(year)


def _seasonal_factor(moment: datetime, amplitude: float) -> float:
    days = 366 if calendar.isleap(moment.year) else 365
    offset = moment.timetuple().tm_yday - _SEASONAL_PEAK_DAY
    return 1.0 + amplitude * math.cos(math.tau * offset / days)


def _household_shape(moment: datetime) -> float:
    shape = (
        _WEEKEND_HOUSEHOLD_SHAPE
        if moment.weekday() >= 5
        else _WEEKDAY_HOUSEHOLD_SHAPE
    )
    return shape[moment.hour] * _seasonal_factor(
        moment, _HOUSEHOLD_SEASONAL_AMPLITUDE
    )


def _hot_water_shape(moment: datetime) -> float:
    return _HOT_WATER_SHAPE[moment.hour] * _seasonal_factor(
        moment, _HOT_WATER_SEASONAL_AMPLITUDE
    )


def _smoothed_temperatures(
    timestamps: list[datetime],
    temperatures: list[float],
) -> list[float]:
    """Six-hour trailing mean so heating demand has thermal inertia."""
    lookup = dict(zip(timestamps, temperatures, strict=True))
    smoothed: list[float] = []
    for timestamp in timestamps:
        history = [
            lookup[timestamp - timedelta(hours=offset)]
            for offset in range(TEMPERATURE_SMOOTHING_HOURS)
            if timestamp - timedelta(hours=offset) in lookup
        ]
        smoothed.append(sum(history) / len(history))
    return smoothed


def _scale_component(
    raw: list[float],
    years: list[int],
    year_hours: dict[int, int],
    covered: dict[int, bool],
    target_kwh: float,
) -> list[float]:
    totals: dict[int, float] = defaultdict(float)
    for year, value in zip(years, raw, strict=True):
        totals[year] += value
    factors: dict[int, float] = {}
    for year, total in totals.items():
        if total <= 0.0:
            raise ValueError(
                f"Cannot scale a {target_kwh:,.0f} kWh component across "
                f"{year}: the raw shape sums to zero. Include hours with "
                "heating demand or select a different component."
            )
        target = _target_kwh(target_kwh, year, year_hours, covered)
        factors[year] = target / total
    return [
        value * factors[year]
        for value, year in zip(raw, years, strict=True)
    ]


def _profiles_complete(
    table: pa.Table,
    profiles: tuple[LoadProfileSpec, ...],
) -> bool:
    for profile in profiles:
        column = table[profile.column]
        if column.null_count:
            return False
        if pc.any(pc.fill_null(pc.is_nan(column), False)).as_py():
            return False
        if pc.any(pc.fill_null(pc.is_inf(column), False)).as_py():
            return False
    return True


def _profiles_non_negative(
    table: pa.Table,
    profiles: tuple[LoadProfileSpec, ...],
) -> bool:
    return all(
        not pc.any(pc.less(table[profile.column], 0.0)).as_py()
        for profile in profiles
    )


def _annual_targets_met(
    table: pa.Table,
    profiles: tuple[LoadProfileSpec, ...],
    years: list[int],
    year_hours: dict[int, int],
    covered: dict[int, bool],
) -> bool:
    for profile in profiles:
        sums: dict[int, float] = defaultdict(float)
        for year, value in zip(
            years, table[profile.column].to_pylist(), strict=True
        ):
            sums[year] += value
        for year, total in sums.items():
            target = _target_kwh(
                profile.annual_kwh, year, year_hours, covered
            )
            if abs(total - target) > _ANNUAL_TOLERANCE_KWH:
                return False
    return True