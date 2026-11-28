"""Calendar features: time encodings, weekends, and Finnish holidays."""

from __future__ import annotations

import math
from datetime import date
from typing import Any

import holidays
import pyarrow as pa
import pyarrow.compute as pc

from fi_energy.datasets.alignment import TIMESTAMP_COLUMN
from fi_energy.features.group import FeatureGroup
from fi_energy.features.spec import FeatureSpec

__all__ = ["CalendarFeatureGroup"]

CALENDAR_GROUP = "calendar"

LOCAL_TIMEZONE = "Europe/Helsinki"
HOLIDAY_COUNTRY = "FI"

SIN_HOUR = "sin_hour"
COS_HOUR = "cos_hour"
SIN_MONTH = "sin_month"
COS_MONTH = "cos_month"
IS_WEEKEND = "is_weekend"
IS_FI_PUBLIC_HOLIDAY = "is_fi_public_holiday"

_FLOAT32 = pa.float32()
_FLOAT64 = pa.float64()
_HOURS_PER_DAY = 24.0
_MONTHS_PER_YEAR = 12.0
_WEEKEND_START = 5  # Monday is 0 in pyarrow's day_of_week, so Saturday is 5.


class CalendarFeatureGroup(FeatureGroup):
    """
    Encodes the position in the day, month, week, and Finnish holiday calendar.

    Calendar features are derived from the UTC timestamp after converting it to
    ``Europe/Helsinki`` local time: hourly demand and day-ahead prices follow
    local wall-clock time, and Finnish public holidays are local calendar days.
    The conversion is unambiguous in the UTC-to-local direction even across
    daylight-saving changes.

    ``sin_hour``/``cos_hour`` place the hour of day on a 24-hour circle, and
    ``sin_month``/``cos_month`` place the month on a 12-month circle, so the
    model sees December as adjacent to January and 23:00 as adjacent to 00:00.
    January and midnight sit at angle zero. ``is_weekend`` is 1.0 on Saturday
    and Sunday; ``is_fi_public_holiday`` is 1.0 on a holiday from the
    ``holidays`` package for country ``FI``.
    """

    name = CALENDAR_GROUP
    specs = (
        FeatureSpec(
            name=SIN_HOUR,
            inputs=(TIMESTAMP_COLUMN,),
            description=(
                "Sine of the local hour of day on a 24-hour cycle; zero at "
                "midnight, 1 at 06:00, 0 at noon, -1 at 18:00."
            ),
        ),
        FeatureSpec(
            name=COS_HOUR,
            inputs=(TIMESTAMP_COLUMN,),
            description=(
                "Cosine of the local hour of day on a 24-hour cycle; 1 at "
                "midnight, 0 at 06:00, -1 at noon, 0 at 18:00."
            ),
        ),
        FeatureSpec(
            name=SIN_MONTH,
            inputs=(TIMESTAMP_COLUMN,),
            description=(
                "Sine of the local month on a 12-month cycle; January is at "
                "angle zero."
            ),
        ),
        FeatureSpec(
            name=COS_MONTH,
            inputs=(TIMESTAMP_COLUMN,),
            description=(
                "Cosine of the local month on a 12-month cycle; 1 in "
                "January, -1 in July."
            ),
        ),
        FeatureSpec(
            name=IS_WEEKEND,
            inputs=(TIMESTAMP_COLUMN,),
            description=(
                "1.0 on Saturday or Sunday in Finnish local time, else 0.0."
            ),
        ),
        FeatureSpec(
            name=IS_FI_PUBLIC_HOLIDAY,
            inputs=(TIMESTAMP_COLUMN,),
            description=(
                "1.0 on a Finnish public holiday (the holidays package for "
                "country FI) in local time, else 0.0."
            ),
        ),
    )

    def validate_inputs(self, table: pa.Table) -> None:
        super().validate_inputs(table)
        dtype = table.schema.field(TIMESTAMP_COLUMN).type
        if not pa.types.is_timestamp(dtype) or dtype.tz != "UTC":
            raise ValueError(
                f"Column {TIMESTAMP_COLUMN!r} must be a UTC timestamp, got "
                f"{dtype}."
            )
        if table[TIMESTAMP_COLUMN].null_count:
            raise ValueError(
                f"Column {TIMESTAMP_COLUMN!r} contains null timestamps."
            )

    def _compute(self, table: pa.Table) -> pa.Table:
        local = _local_timestamps(table)
        months = pc.subtract(pc.month(local), 1)
        features = (
            (SIN_HOUR, _sin(pc.hour(local), _HOURS_PER_DAY)),
            (COS_HOUR, _cos(pc.hour(local), _HOURS_PER_DAY)),
            (SIN_MONTH, _sin(months, _MONTHS_PER_YEAR)),
            (COS_MONTH, _cos(months, _MONTHS_PER_YEAR)),
            (
                IS_WEEKEND,
                pc.cast(
                    pc.greater_equal(
                        pc.day_of_week(local),
                        _WEEKEND_START,
                    ),
                    _FLOAT32,
                ),
            ),
            (
                IS_FI_PUBLIC_HOLIDAY,
                _holiday_flags([value.date() for value in local.to_pylist()]),
            ),
        )
        for name, values in features:
            table = table.append_column(name, pc.cast(values, _FLOAT32))
        return table

    def diagnostics(self, table: pa.Table) -> dict[str, Any]:
        """
        Summarises the local calendar and the hours it flags.

        Call it on the raw source table before building features.
        """
        self._require_columns(table, (TIMESTAMP_COLUMN,))
        local = _local_timestamps(table)
        dates = [value.date() for value in local.to_pylist()]
        calendar = _fi_holidays(dates)
        weekend = pc.greater_equal(pc.day_of_week(local), _WEEKEND_START)
        return {
            "weekend_hours": int(
                pc.sum(pc.cast(weekend, pa.int64())).as_py() or 0
            ),
            "holiday_hours": sum(1 for day in dates if day in calendar),
            "distinct_holidays": len(
                {day for day in dates if day in calendar}
            ),
        }


def _local_timestamps(table: pa.Table) -> pa.ChunkedArray:
    dtype = table.schema.field(TIMESTAMP_COLUMN).type
    return pc.cast(
        table[TIMESTAMP_COLUMN],
        pa.timestamp(dtype.unit, tz=LOCAL_TIMEZONE),
    )


def _sin(values: pa.Array | pa.ChunkedArray, period: float) -> pa.Array:
    return pc.sin(_angle(values, period))


def _cos(values: pa.Array | pa.ChunkedArray, period: float) -> pa.Array:
    return pc.cos(_angle(values, period))


def _angle(values: pa.Array | pa.ChunkedArray, period: float) -> pa.Array:
    return pc.multiply(
        pc.cast(values, _FLOAT64),
        pa.scalar(2.0 * math.pi / period, type=_FLOAT64),
    )


def _fi_holidays(dates: list[date]) -> holidays.HolidayBase:
    years = sorted({value.year for value in dates})
    if not years:
        return holidays.HolidayBase()
    return holidays.country_holidays(HOLIDAY_COUNTRY, years=years)


def _holiday_flags(dates: list[date]) -> pa.Array:
    calendar = _fi_holidays(dates)
    return pa.array(
        [1.0 if value in calendar else 0.0 for value in dates],
        type=_FLOAT32,
    )