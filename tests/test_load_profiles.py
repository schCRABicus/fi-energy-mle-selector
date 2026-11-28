from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

import pyarrow as pa
import pyarrow.compute as pc
import pytest

from fi_energy.profiles import (
    APARTMENT_2500,
    DEFAULT_PROFILES,
    DETACHED_HOUSE_18000,
    HOURLY_KWH,
    LoadProfileSpec,
    build_load_profile_table,
    profile_annual_totals,
    profile_table,
)

START_2025 = datetime(2025, 1, 1, tzinfo=UTC)
START_2024 = datetime(2024, 1, 1, tzinfo=UTC)
LOCAL = ZoneInfo("Europe/Helsinki")


def make_source(
    start: datetime,
    rows: int,
    temperature: float | callable = 5.0,
) -> pa.Table:
    timestamps = [start + timedelta(hours=index) for index in range(rows)]
    if callable(temperature):
        temperatures = [temperature(timestamp) for timestamp in timestamps]
    else:
        temperatures = [temperature] * rows
    return pa.Table.from_arrays(
        [
            pa.array(timestamps, type=pa.timestamp("us", tz="UTC")),
            pa.array(temperatures, type=pa.float32()),
        ],
        names=["timestamp_utc", "helsinki_temperature_c"],
    )


def seasonal_temperature(timestamp: datetime) -> float:
    day = timestamp.timetuple().tm_yday
    days = 366 if timestamp.year % 4 == 0 else 365
    return 5.0 - 14.0 * math.cos(math.tau * (day - 15) / days)


def half_year_2025() -> pa.Table:
    start = datetime(2025, 1, 1, tzinfo=UTC)
    end = datetime(2025, 7, 1, tzinfo=UTC)
    return make_source(start, int((end - start) / timedelta(hours=1)))


def test_builds_both_profiles_with_expected_columns_and_annual_energy():
    build = build_load_profile_table(
        make_source(START_2025, 365 * 24, seasonal_temperature)
    )

    assert build.table.column_names == [
        "timestamp_utc",
        APARTMENT_2500.column,
        DETACHED_HOUSE_18000.column,
    ]
    assert build.table.num_rows == 365 * 24
    totals = profile_annual_totals(build)
    assert totals["apartment_2500"][2025] == pytest.approx(2_500.0, abs=0.5)
    assert totals["detached_house_18000"][2025] == pytest.approx(
        18_000.0, abs=0.5
    )


def test_profiles_contain_no_nulls_nans_infinities_or_negatives():
    build = build_load_profile_table(
        make_source(START_2025, 365 * 24, seasonal_temperature)
    )

    for profile in DEFAULT_PROFILES:
        column = build.table[profile.column]
        assert column.null_count == 0
        assert not pc.any(pc.fill_null(pc.is_nan(column), False)).as_py()
        assert not pc.any(pc.fill_null(pc.is_inf(column), False)).as_py()
        assert not pc.any(pc.less(column, 0.0)).as_py()


def test_apartment_has_a_strong_evening_peak():
    build = build_load_profile_table(make_source(START_2025, 365 * 24))
    rows = build.table.to_pylist()
    evening = [
        row[APARTMENT_2500.column]
        for row in rows
        if 17 <= row["timestamp_utc"].astimezone(LOCAL).hour <= 21
    ]
    night = [
        row[APARTMENT_2500.column]
        for row in rows
        if row["timestamp_utc"].astimezone(LOCAL).hour <= 5
    ]

    assert sum(evening) / len(evening) > 1.8 * sum(night) / len(night)


def test_electrically_heated_house_responds_to_outdoor_temperature():
    build = build_load_profile_table(
        make_source(START_2025, 365 * 24, seasonal_temperature)
    )
    rows = build.table.to_pylist()

    def mean(column: str, month: int) -> float:
        values = [
            row[column]
            for row in rows
            if row["timestamp_utc"].month == month
        ]
        return sum(values) / len(values)

    house_winter = mean(DETACHED_HOUSE_18000.column, 1)
    house_summer = mean(DETACHED_HOUSE_18000.column, 7)
    apartment_winter = mean(APARTMENT_2500.column, 1)
    apartment_summer = mean(APARTMENT_2500.column, 7)

    assert house_winter > 3.0 * house_summer
    assert 1.0 < apartment_winter / apartment_summer < 1.5


def test_each_calendar_year_is_normalized_independently():
    source = make_source(START_2025, 2 * 365 * 24, seasonal_temperature)
    build = build_load_profile_table(source)

    totals = profile_annual_totals(build)
    assert totals["apartment_2500"] == pytest.approx(
        {2025: 2_500.0, 2026: 2_500.0}, abs=0.5
    )
    assert totals["detached_house_18000"] == pytest.approx(
        {2025: 18_000.0, 2026: 18_000.0}, abs=0.5
    )


def test_interior_gaps_still_total_the_full_annual_energy():
    source = make_source(START_2025, 365 * 24, seasonal_temperature)
    timestamps = source["timestamp_utc"].to_pylist()
    keep = [
        index
        for index, timestamp in enumerate(timestamps)
        if not (
            datetime(2025, 3, 1, tzinfo=UTC)
            <= timestamp
            < datetime(2025, 3, 3, tzinfo=UTC)
        )
    ]
    source = source.take(pa.array(keep, type=pa.int64()))

    build = build_load_profile_table(source)

    totals = profile_annual_totals(build)
    assert build.table.num_rows == 365 * 24 - 48
    assert totals["apartment_2500"][2025] == pytest.approx(2_500.0, abs=0.5)
    assert totals["detached_house_18000"][2025] == pytest.approx(
        18_000.0, abs=0.5
    )


def test_partial_year_is_scaled_by_its_share_of_the_year():
    source = half_year_2025()
    build = build_load_profile_table(source)

    fraction = source.num_rows / (365 * 24)
    totals = profile_annual_totals(build)
    assert totals["apartment_2500"][2025] == pytest.approx(
        2_500.0 * fraction, abs=0.5
    )
    assert totals["detached_house_18000"][2025] == pytest.approx(
        18_000.0 * fraction, abs=0.5
    )


def test_leap_year_uses_8784_hours():
    build = build_load_profile_table(
        make_source(START_2024, 366 * 24, seasonal_temperature)
    )

    totals = profile_annual_totals(build)
    assert build.table.num_rows == 8_784
    assert totals["detached_house_18000"][2024] == pytest.approx(
        18_000.0, abs=0.5
    )


def test_profile_table_extracts_hourly_kwh():
    build = build_load_profile_table(make_source(START_2025, 48))

    apartment = profile_table(build, "apartment_2500")
    assert apartment.column_names == ["timestamp_utc", HOURLY_KWH]
    assert apartment[HOURLY_KWH].to_pylist() == build.table[
        APARTMENT_2500.column
    ].to_pylist()


def test_unknown_profile_key_is_rejected():
    build = build_load_profile_table(make_source(START_2025, 48))

    with pytest.raises(KeyError, match="Unknown"):
        profile_table(build, "unknown")


def test_missing_temperature_column_is_rejected():
    source = make_source(START_2025, 48).drop_columns(
        ["helsinki_temperature_c"]
    )

    with pytest.raises(ValueError, match="helsinki_temperature_c"):
        build_load_profile_table(source)


def test_null_temperature_is_rejected():
    source = make_source(START_2025, 48)
    values = source["helsinki_temperature_c"].to_pylist()
    values[10] = None
    source = source.set_column(
        1,
        "helsinki_temperature_c",
        pa.array(values, type=pa.float32()),
    )

    with pytest.raises(ValueError, match="null"):
        build_load_profile_table(source)


def test_unsorted_timestamps_are_rejected():
    source = make_source(START_2025, 48)
    timestamps = source["timestamp_utc"].to_pylist()
    timestamps.reverse()
    source = source.set_column(
        0,
        "timestamp_utc",
        pa.array(timestamps, type=pa.timestamp("us", tz="UTC")),
    )

    with pytest.raises(ValueError, match="ascending"):
        build_load_profile_table(source)


def test_source_table_is_not_modified():
    source = make_source(START_2025, 48, seasonal_temperature)
    original = source.to_pydict()

    build_load_profile_table(source)

    assert source.to_pydict() == original
    assert build_load_profile_table(source).table is not source


def test_profile_spec_rejects_components_that_do_not_sum():
    with pytest.raises(ValueError, match="must sum"):
        LoadProfileSpec(
            key="broken",
            label="Broken",
            annual_kwh=1_000.0,
            household_kwh=500.0,
            hot_water_kwh=400.0,
            space_heating_kwh=0.0,
            description="Components do not add up.",
        )


def test_profile_spec_rejects_invalid_key_and_zero_annual_energy():
    with pytest.raises(ValueError, match="identifier"):
        LoadProfileSpec(
            key="Bad Key",
            label="Bad",
            annual_kwh=1.0,
            household_kwh=1.0,
            hot_water_kwh=0.0,
            space_heating_kwh=0.0,
            description="Bad key.",
        )
    with pytest.raises(ValueError, match="positive"):
        LoadProfileSpec(
            key="zero",
            label="Zero",
            annual_kwh=0.0,
            household_kwh=0.0,
            hot_water_kwh=0.0,
            space_heating_kwh=0.0,
            description="Zero annual energy.",
        )