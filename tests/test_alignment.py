from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

import pyarrow as pa
import pytest

from fi_energy.datasets.alignment import (
    align_hourly,
    drop_incomplete_boundary_rows,
    hourly_index,
    interpolate_short_gaps,
    trailing_window,
    validate_hourly_table,
    window_label,
)

START = datetime(2024, 1, 1, tzinfo=UTC)
END = datetime(2024, 1, 1, 4, tzinfo=UTC)
UTC_TIMESTAMP = pa.timestamp("us", tz="UTC")


def make_table(
    rows: list[tuple[datetime, float | int | None]],
    *,
    column: str = "value",
    tz: str | None = "UTC",
) -> pa.Table:
    schema = pa.schema([
        pa.field("timestamp_utc", pa.timestamp("us", tz=tz)),
        pa.field(column, pa.float32()),
    ])
    return pa.Table.from_pydict(
        {
            "timestamp_utc": [timestamp for timestamp, _ in rows],
            column: [value for _, value in rows],
        },
        schema=schema,
    )


def test_trailing_window_ends_on_the_current_utc_hour():
    now = datetime(2026, 5, 17, 13, 42, 7, tzinfo=UTC)

    start, end = trailing_window(now)

    assert end == datetime(2026, 5, 17, 13, tzinfo=UTC)
    assert start == datetime(2023, 5, 17, 13, tzinfo=UTC)


def test_trailing_window_handles_a_leap_day():
    now = datetime(2024, 2, 29, 12, 34, tzinfo=UTC)

    start, end = trailing_window(now)

    assert end == datetime(2024, 2, 29, 12, tzinfo=UTC)
    assert start == datetime(2021, 2, 28, 12, tzinfo=UTC)


def test_trailing_window_normalizes_the_timezone():
    east_two = timezone(timedelta(hours=2))

    start, end = trailing_window(datetime(2026, 5, 17, 15, 42, tzinfo=east_two))

    assert end == datetime(2026, 5, 17, 13, tzinfo=UTC)
    assert start == datetime(2023, 5, 17, 13, tzinfo=UTC)


def test_trailing_window_rejects_naive_now():
    with pytest.raises(ValueError, match="timezone-aware"):
        trailing_window(datetime(2026, 5, 17, 13, 42))


def test_trailing_window_rejects_non_positive_years():
    with pytest.raises(ValueError, match="at least 1"):
        trailing_window(years=0)


def test_window_label_describes_calendar_year_windows():
    start, end = trailing_window(datetime(2026, 11, 26, 4, 31, tzinfo=UTC))

    assert window_label(start, end) == "3y"
    assert window_label(datetime(2024, 1, 1, tzinfo=UTC),
                        datetime(2026, 1, 1, tzinfo=UTC)) == "2y"


def test_window_label_falls_back_to_exact_hours():
    one_day = datetime(2024, 3, 1, tzinfo=UTC)
    assert window_label(one_day, one_day + timedelta(days=1)) == "24h"


def test_window_label_is_leap_day_aware():
    start, end = trailing_window(datetime(2024, 2, 29, 12, tzinfo=UTC))

    assert (start, end) == (
        datetime(2021, 2, 28, 12, tzinfo=UTC),
        datetime(2024, 2, 29, 12, tzinfo=UTC),
    )
    assert window_label(start, end) == "3y"


def test_hourly_index_covers_the_half_open_window():
    assert hourly_index(START, END) == [
        START,
        START + timedelta(hours=1),
        START + timedelta(hours=2),
        START + timedelta(hours=3),
    ]


@pytest.mark.parametrize(
    ("start", "end", "message"),
    [
        (START + timedelta(minutes=30), END, "exact UTC hour"),
        (START, END + timedelta(minutes=15), "exact UTC hour"),
        (START, START, "must be before"),
    ],
)
def test_hourly_index_rejects_invalid_bounds(start, end, message):
    with pytest.raises(ValueError, match=message):
        hourly_index(start, end)


def test_align_hourly_outer_joins_on_the_complete_grid():
    first = make_table(
        [(START, 1.0), (START + timedelta(hours=2), 3.0)],
        column="first",
    )
    second = make_table(
        [(START + timedelta(hours=1), 2.0), (START + timedelta(hours=3), 4.0)],
        column="second",
    )

    table = align_hourly({"first": first, "second": second}, start=START, end=END)

    assert table.schema.field("timestamp_utc").type == UTC_TIMESTAMP
    assert table.schema.field("first").type == pa.float32()
    assert table.schema.field("first").nullable
    assert table["timestamp_utc"].to_pylist() == hourly_index(START, END)
    assert table["first"].to_pylist() == [1.0, None, 3.0, None]
    assert table["second"].to_pylist() == [None, 2.0, None, 4.0]


def test_align_hourly_keeps_columns_from_an_empty_table():
    empty = pa.Table.from_pydict(
        {"timestamp_utc": [], "missing": []},
        schema=pa.schema([
            pa.field("timestamp_utc", UTC_TIMESTAMP),
            pa.field("missing", pa.float32()),
        ]),
    )
    present = make_table([(START, 1.0)], column="present")

    table = align_hourly({"empty": empty, "present": present}, start=START, end=END)

    assert table.column_names == ["timestamp_utc", "missing", "present"]
    assert table["missing"].to_pylist() == [None, None, None, None]
    assert table["present"].to_pylist() == [1.0, None, None, None]


def test_align_hourly_normalizes_non_utc_timestamps():
    east_two = timezone(timedelta(hours=2))
    table = make_table(
        [(datetime(2024, 1, 1, 2, tzinfo=east_two), 7.0)],
    )

    aligned = align_hourly({"source": table}, start=START, end=END)

    assert aligned["timestamp_utc"].to_pylist() == hourly_index(START, END)
    assert aligned["value"].to_pylist() == [7.0, None, None, None]


def test_align_hourly_rejects_duplicate_value_columns():
    first = make_table([(START, 1.0)], column="value")
    second = make_table([(START + timedelta(hours=1), 2.0)], column="value")

    with pytest.raises(ValueError, match="more than one table"):
        align_hourly({"first": first, "second": second}, start=START, end=END)


def test_align_hourly_rejects_off_grid_timestamps():
    table = make_table([(START + timedelta(minutes=30), 1.0)])

    with pytest.raises(ValueError, match="hourly grid"):
        align_hourly({"source": table}, start=START, end=END)


def test_align_hourly_rejects_out_of_window_timestamps():
    table = make_table([(END, 1.0)])

    with pytest.raises(ValueError, match="outside"):
        align_hourly({"source": table}, start=START, end=END)


def test_align_hourly_rejects_duplicate_timestamps():
    table = make_table([(START, 1.0), (START, 2.0)])

    with pytest.raises(ValueError, match="duplicate timestamp"):
        align_hourly({"source": table}, start=START, end=END)


def test_validate_hourly_table_accepts_sparse_but_aligned_rows():
    table = make_table([
        (START, 1.0),
        (START + timedelta(hours=2), None),
    ])

    validate_hourly_table(table, start=START, end=END, label="sparse")


def test_validate_hourly_table_rejects_naive_timestamps():
    table = make_table([(datetime(2024, 1, 1), 1.0)], tz=None)

    with pytest.raises(ValueError, match="must be timestamp"):
        validate_hourly_table(table, start=START, end=END)


def test_validate_hourly_table_rejects_a_missing_timestamp_column():
    table = pa.table({"value": pa.array([1.0], type=pa.float32())})

    with pytest.raises(ValueError, match="no 'timestamp_utc' column"):
        validate_hourly_table(table, start=START, end=END)


def test_validate_hourly_table_rejects_a_non_utc_timestamp_type():
    table = pa.Table.from_pydict(
        {"timestamp_utc": [START], "value": [1.0]},
        schema=pa.schema([
            pa.field("timestamp_utc", pa.timestamp("us", tz="Europe/Helsinki")),
            pa.field("value", pa.float32()),
        ]),
    )

    with pytest.raises(ValueError, match="must be"):
        validate_hourly_table(table, start=START, end=END)


def test_interpolate_short_gaps_fills_one_and_two_hour_runs():
    table = make_table([
        (START, 1.0),
        (START + timedelta(hours=1), None),
        (START + timedelta(hours=2), 3.0),
        (START + timedelta(hours=3), None),
        (START + timedelta(hours=4), None),
        (START + timedelta(hours=5), 6.0),
        (START + timedelta(hours=6), None),
    ])

    filled = interpolate_short_gaps(table, max_gap=2)

    assert filled["value"].to_pylist() == [
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        None,
    ]
    assert filled.schema == table.schema


def test_interpolate_short_gaps_leaves_longer_runs():
    table = make_table([
        (START, 1.0),
        (START + timedelta(hours=1), None),
        (START + timedelta(hours=2), None),
        (START + timedelta(hours=3), None),
        (START + timedelta(hours=4), 5.0),
    ])

    filled = interpolate_short_gaps(table, max_gap=2)

    assert filled["value"].to_pylist() == [1.0, None, None, None, 5.0]


def test_interpolate_short_gaps_respects_max_gap():
    table = make_table([
        (START, 1.0),
        (START + timedelta(hours=1), None),
        (START + timedelta(hours=2), None),
        (START + timedelta(hours=3), 4.0),
    ])

    assert interpolate_short_gaps(table, max_gap=0)["value"].to_pylist() == [
        1.0, None, None, 4.0,
    ]
    assert interpolate_short_gaps(table, max_gap=1)["value"].to_pylist() == [
        1.0, None, None, 4.0,
    ]
    assert interpolate_short_gaps(table, max_gap=2)["value"].to_pylist() == [
        1.0, pytest.approx(2.0), pytest.approx(3.0), 4.0,
    ]


def test_interpolate_short_gaps_keeps_leading_nulls():
    table = make_table([
        (START, None),
        (START + timedelta(hours=1), 2.0),
        (START + timedelta(hours=2), 4.0),
    ])

    filled = interpolate_short_gaps(table, max_gap=2)

    assert filled["value"].to_pylist() == [None, 2.0, 4.0]


def test_interpolate_short_gaps_can_target_one_column():
    table = pa.Table.from_pydict(
        {
            "timestamp_utc": [START, START + timedelta(hours=1), START + timedelta(hours=2)],
            "first": [1.0, None, 3.0],
            "second": [10.0, None, 30.0],
        },
        schema=pa.schema([
            pa.field("timestamp_utc", UTC_TIMESTAMP),
            pa.field("first", pa.float32()),
            pa.field("second", pa.float32()),
        ]),
    )

    filled = interpolate_short_gaps(table, columns=["first"])

    assert filled["first"].to_pylist() == [1.0, 2.0, 3.0]
    assert filled["second"].to_pylist() == [10.0, None, 30.0]


def test_interpolate_short_gaps_rejects_unknown_column():
    table = make_table([(START, 1.0)])

    with pytest.raises(ValueError, match="Unknown value column"):
        interpolate_short_gaps(table, columns=["missing"])


def test_interpolate_short_gaps_preserves_metadata():
    table = make_table([(START, 1.0)]).replace_schema_metadata({b"k": b"v"})

    filled = interpolate_short_gaps(table)

    assert filled.schema.metadata == {b"k": b"v"}


def test_drop_incomplete_boundary_rows_trims_edges_not_the_interior():
    table = make_table([
        (START, None),
        (START + timedelta(hours=1), 1.0),
        (START + timedelta(hours=2), None),
        (START + timedelta(hours=3), 3.0),
        (START + timedelta(hours=4), None),
    ])

    trimmed = drop_incomplete_boundary_rows(table)

    assert trimmed["timestamp_utc"].to_pylist() == [
        START + timedelta(hours=1),
        START + timedelta(hours=2),
        START + timedelta(hours=3),
    ]
    assert trimmed["value"].to_pylist() == [1.0, None, 3.0]


def test_drop_incomplete_boundary_rows_requires_every_column():
    table = pa.Table.from_pydict(
        {
            "timestamp_utc": [
                START,
                START + timedelta(hours=1),
                START + timedelta(hours=2),
                START + timedelta(hours=3),
            ],
            "first": [None, 1.0, 2.0, 3.0],
            "second": [10.0, 20.0, 30.0, None],
        },
        schema=pa.schema([
            pa.field("timestamp_utc", UTC_TIMESTAMP),
            pa.field("first", pa.float32()),
            pa.field("second", pa.float32()),
        ]),
    )

    trimmed = drop_incomplete_boundary_rows(table)

    assert trimmed["timestamp_utc"].to_pylist() == [
        START + timedelta(hours=1),
        START + timedelta(hours=2),
    ]


def test_drop_incomplete_boundary_rows_handles_an_all_null_column():
    table = make_table([
        (START, None),
        (START + timedelta(hours=1), None),
    ])

    assert drop_incomplete_boundary_rows(table).num_rows == 0