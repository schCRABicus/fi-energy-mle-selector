from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

import pyarrow as pa
import pytest

from fi_energy.features import (
    CalendarFeatureGroup,
    FeatureCatalog,
    build_feature_table,
    describe_features,
)
from fi_energy.features.groups.calendar import (
    CALENDAR_GROUP,
    COS_HOUR,
    COS_MONTH,
    IS_FI_PUBLIC_HOLIDAY,
    IS_WEEKEND,
    SIN_HOUR,
    SIN_MONTH,
)

UTC_TIMESTAMP = pa.timestamp("us", tz="UTC")
FEATURE_COLUMNS = [
    SIN_HOUR,
    COS_HOUR,
    SIN_MONTH,
    COS_MONTH,
    IS_WEEKEND,
    IS_FI_PUBLIC_HOLIDAY,
]
GROUP = CalendarFeatureGroup()
CALENDAR_ONLY_CATALOG = FeatureCatalog((GROUP,))


def make_table(
    timestamps: list[datetime | None],
    *,
    nullable_timestamp: bool = False,
) -> pa.Table:
    schema = pa.schema([
        pa.field(
            "timestamp_utc",
            UTC_TIMESTAMP,
            nullable=nullable_timestamp,
        )
    ])
    return pa.Table.from_pydict(
        {"timestamp_utc": timestamps},
        schema=schema,
    )


def at(*parts: int) -> datetime:
    return datetime(*parts, tzinfo=UTC)


def test_compute_appends_calendar_features_in_declaration_order():
    result = GROUP.compute(make_table([at(2024, 1, 1, 12)]))

    assert result.column_names == ["timestamp_utc", *FEATURE_COLUMNS]
    for name in FEATURE_COLUMNS:
        field = result.schema.field(name)
        assert field.type == pa.float32()
        assert field.nullable


def test_hour_encodings_use_finnish_local_time():
    # 10:00 UTC is noon in winter (UTC+2); 21:00 UTC is midnight in summer
    # (UTC+3) on the following local day.
    result = GROUP.compute(
        make_table([at(2024, 1, 1, 10), at(2024, 7, 1, 21)])
    )

    assert result[SIN_HOUR].to_pylist() == pytest.approx([0.0, 0.0], abs=1e-6)
    assert result[COS_HOUR].to_pylist() == pytest.approx([-1.0, 1.0], abs=1e-6)


def test_month_encodings_use_the_local_calendar():
    # 22:00 UTC on 31 January is midnight on 1 February in Helsinki.
    result = GROUP.compute(
        make_table([at(2024, 1, 31, 22), at(2024, 12, 15, 12)])
    )

    expected_sin = [
        math.sin(2 * math.pi / 12),
        math.sin(2 * math.pi * 11 / 12),
    ]
    expected_cos = [
        math.cos(2 * math.pi / 12),
        math.cos(2 * math.pi * 11 / 12),
    ]
    assert result[SIN_MONTH].to_pylist() == pytest.approx(
        expected_sin,
        abs=1e-6,
    )
    assert result[COS_MONTH].to_pylist() == pytest.approx(
        expected_cos,
        abs=1e-6,
    )


def test_hour_and_month_encodings_lie_on_the_unit_circle():
    result = GROUP.compute(
        make_table([
            at(2024, 3, 1, 0),
            at(2024, 3, 1, 7),
            at(2024, 9, 30, 23),
        ])
    )

    hour_norm = [
        sin**2 + cos**2
        for sin, cos in zip(
            result[SIN_HOUR].to_pylist(),
            result[COS_HOUR].to_pylist(),
        )
    ]
    month_norm = [
        sin**2 + cos**2
        for sin, cos in zip(
            result[SIN_MONTH].to_pylist(),
            result[COS_MONTH].to_pylist(),
        )
    ]
    assert hour_norm == pytest.approx([1.0, 1.0, 1.0], abs=1e-6)
    assert month_norm == pytest.approx([1.0, 1.0, 1.0], abs=1e-6)


def test_weekend_flag_uses_local_dates():
    # 23:00 UTC on Friday is already Saturday in Helsinki.
    result = GROUP.compute(
        make_table([
            at(2024, 1, 5, 12),   # Friday
            at(2024, 1, 6, 12),   # Saturday
            at(2024, 1, 7, 12),   # Sunday
            at(2024, 1, 5, 23),   # Saturday 01:00 local
        ])
    )

    assert result[IS_WEEKEND].to_pylist() == [0.0, 1.0, 1.0, 1.0]


def test_holiday_flag_uses_finnish_local_dates():
    result = GROUP.compute(
        make_table([
            at(2024, 12, 5, 23),   # 6 December 01:00 local, Independence Day
            at(2024, 12, 6, 22),   # 7 December 00:00 local, not a holiday
            at(2024, 1, 2, 12),    # ordinary Tuesday
            at(2024, 6, 20, 22),   # 21 June 01:00 local, Midsummer Eve
        ])
    )

    assert result[IS_FI_PUBLIC_HOLIDAY].to_pylist() == [1.0, 0.0, 0.0, 1.0]


def test_compute_preserves_source_columns_and_metadata():
    table = make_table([at(2024, 1, 1, 12)]).replace_schema_metadata(
        {b"k": b"v"}
    )

    result = GROUP.compute(table)

    assert result.column_names[:1] == table.column_names
    assert result.schema.metadata == {b"k": b"v"}


def test_compute_rejects_missing_timestamp_column():
    table = pa.table({"other": [1]})

    with pytest.raises(ValueError, match="missing required column"):
        GROUP.compute(table)


def test_compute_rejects_null_timestamps():
    table = make_table([None], nullable_timestamp=True)

    with pytest.raises(ValueError, match="null timestamps"):
        GROUP.compute(table)


@pytest.mark.parametrize(
    "timestamps",
    [
        pa.array(
            [datetime(2024, 1, 1, 12, tzinfo=ZoneInfo("Europe/Helsinki"))],
            type=pa.timestamp("us", tz="Europe/Helsinki"),
        ),
        pa.array([datetime(2024, 1, 1, 12)], type=pa.timestamp("us")),
        pa.array(["2024-01-01"], type=pa.string()),
    ],
    ids=["local timezone", "naive", "string"],
)
def test_compute_rejects_non_utc_timestamps(timestamps):
    table = pa.table({"timestamp_utc": timestamps})

    with pytest.raises(ValueError, match="must be a UTC timestamp"):
        GROUP.compute(table)


def test_compute_rejects_non_table():
    with pytest.raises(TypeError, match="pyarrow Table"):
        GROUP.compute({})  # type: ignore[arg-type]


def test_compute_rejects_existing_feature_columns():
    table = make_table([at(2024, 1, 1, 12)]).append_column(
        SIN_HOUR,
        pa.array([0.0], type=pa.float32()),
    )

    with pytest.raises(ValueError, match="already has feature column"):
        GROUP.compute(table)


def test_compute_handles_an_empty_table():
    result = GROUP.compute(make_table([]))

    assert result.num_rows == 0
    for name in FEATURE_COLUMNS:
        assert result.schema.field(name).type == pa.float32()


def test_diagnostics_count_weekend_and_holiday_hours():
    table = make_table([
        at(2024, 1, 13, 12),   # Saturday
        at(2024, 1, 14, 12),   # Sunday
        at(2024, 1, 15, 12),   # Monday
        at(2024, 12, 6, 12),   # Independence Day, Friday
    ])

    diagnostics = GROUP.diagnostics(table)

    assert diagnostics == {
        "weekend_hours": 2,
        "holiday_hours": 1,
        "distinct_holidays": 1,
    }


def test_build_feature_table_with_the_calendar_group():
    timestamps = [
        at(2024, 12, 5, 22) + timedelta(hours=index)
        for index in range(24)
    ]
    table = make_table(timestamps)

    result = build_feature_table(table, catalog=CALENDAR_ONLY_CATALOG)
    description = describe_features(
        table,
        result,
        catalog=CALENDAR_ONLY_CATALOG,
    )

    assert result.column_names == ["timestamp_utc", *FEATURE_COLUMNS]
    assert description["groups"] == [CALENDAR_GROUP]
    assert description["feature_columns"] == FEATURE_COLUMNS
    assert [entry["group"] for entry in description["features"]] == [
        CALENDAR_GROUP
    ] * len(FEATURE_COLUMNS)
    assert description["diagnostics"][CALENDAR_GROUP] == {
        "weekend_hours": 0,
        "holiday_hours": 24,
        "distinct_holidays": 1,
    }
    assert description["null_values"] == dict.fromkeys(FEATURE_COLUMNS, 0)