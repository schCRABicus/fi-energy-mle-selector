from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pyarrow as pa
import pyarrow.compute as pc
import pytest

from fi_energy.features import (
    FEATURE_COLUMNS,
    RAW_FEATURE_COLUMNS,
    SOURCE_TARGET_COLUMN,
    TARGET_COLUMN,
    build_training_dataset,
    feature_statistics,
)

START = datetime(2024, 1, 1, tzinfo=UTC)
ROWS = 120


def make_source(
    *,
    rows: int = ROWS,
    null_at: int | None = None,
    nan_at: int | None = None,
    start: datetime = START,
) -> pa.Table:
    timestamps = [start + timedelta(hours=index) for index in range(rows)]
    values: dict[str, list[float | None]] = {
        "wind_mw": [100.0 + index for index in range(rows)],
        "wind_capacity_mw": [5_000.0] * rows,
        "consumption_mw": [10_000.0 + index for index in range(rows)],
        "nuclear_mw": [4_000.0] * rows,
        "hydro_mw": [1_500.0] * rows,
        "se1_flow_mw": [-500.0] * rows,
        "se3_flow_mw": [200.0] * rows,
        "ee_flow_mw": [300.0] * rows,
        "helsinki_temperature_c": [-5.0 + index * 0.05 for index in range(rows)],
        "vaasa_temperature_c": [-3.0] * rows,
        "oulu_temperature_c": [-8.0] * rows,
        "helsinki_radiation_wm2": [0.0] * rows,
        SOURCE_TARGET_COLUMN: [5.0 + index * 0.1 for index in range(rows)],
    }
    if null_at is not None:
        values["wind_mw"][null_at] = None
    if nan_at is not None:
        values["ee_flow_mw"][nan_at] = float("nan")
    arrays = [
        pa.array(timestamps, type=pa.timestamp("us", tz="UTC")),
        *(
            pa.array(values[name], type=pa.float32())
            for name in (*RAW_FEATURE_COLUMNS, SOURCE_TARGET_COLUMN)
        ),
    ]
    names = [
        "timestamp_utc",
        *RAW_FEATURE_COLUMNS,
        SOURCE_TARGET_COLUMN,
    ]
    return pa.Table.from_arrays(arrays, names=names)


def test_dataset_has_24_features_and_a_next_hour_target():
    build = build_training_dataset(make_source())

    assert len(FEATURE_COLUMNS) == 24
    assert build.table.column_names == [
        "timestamp_utc",
        *FEATURE_COLUMNS,
        TARGET_COLUMN,
    ]
    assert build.table.num_columns == 26
    assert TARGET_COLUMN not in build.feature_columns
    assert SOURCE_TARGET_COLUMN not in build.feature_columns


def test_dataset_contains_no_nulls_nans_or_infinities():
    build = build_training_dataset(
        make_source(null_at=80, nan_at=90)
    )

    for name in (*build.feature_columns, build.target_column):
        column = build.table[name]
        assert column.null_count == 0
        assert not pc.any(
            pc.fill_null(pc.is_nan(column), False)
        ).as_py()
        assert not pc.any(
            pc.fill_null(pc.is_inf(column), False)
        ).as_py()


def test_incomplete_rows_are_dropped():
    build = build_training_dataset(
        make_source(null_at=80, nan_at=90)
    )

    # The first 71 rows lack a full 72-hour rolling window, the final row has
    # no next-hour target, and the injected bad rows at 80 and 90 add two more.
    assert build.dropped_rows == 71 + 1 + 2
    assert build.table.num_rows == build.source_rows - build.dropped_rows


def test_target_is_exactly_the_next_hour_spot_price():
    source = make_source()
    build = build_training_dataset(source)

    spot = {
        timestamp: value
        for timestamp, value in zip(
            source["timestamp_utc"].to_pylist(),
            source[SOURCE_TARGET_COLUMN].to_pylist(),
        )
    }
    for timestamp, target in zip(
        build.table["timestamp_utc"].to_pylist(),
        build.table[TARGET_COLUMN].to_pylist(),
    ):
        assert target == pytest.approx(spot[timestamp + timedelta(hours=1)])


def test_dataset_reports_every_leakage_check():
    build = build_training_dataset(make_source())

    assert build.checks == {
        "target_absent_from_features": True,
        "target_absent_from_specs": True,
        "features_ignore_future_rows": True,
        "target_is_next_hour": True,
    }


def test_future_source_changes_do_not_affect_earlier_features():
    source = make_source()
    cut = 90
    future = make_source(rows=ROWS)
    changed: dict[str, list[float | None]] = {
        name: future[name].to_pylist() for name in RAW_FEATURE_COLUMNS
    }
    for name in RAW_FEATURE_COLUMNS:
        for index in range(cut, ROWS):
            changed[name][index] = changed[name][index] * 0 + 1.0
    changed_arrays = [
        future["timestamp_utc"],
        *(
            pa.array(changed[name], type=pa.float32())
            for name in RAW_FEATURE_COLUMNS
        ),
        future[SOURCE_TARGET_COLUMN],
    ]
    changed_source = pa.Table.from_arrays(
        changed_arrays,
        names=["timestamp_utc", *RAW_FEATURE_COLUMNS, SOURCE_TARGET_COLUMN],
    )

    full = build_training_dataset(source).table
    modified = build_training_dataset(changed_source).table

    full_rows = {
        row["timestamp_utc"]: row
        for row in full.to_pylist()
        if row["timestamp_utc"] < START + timedelta(hours=cut)
    }
    modified_rows = {
        row["timestamp_utc"]: row
        for row in modified.to_pylist()
        if row["timestamp_utc"] < START + timedelta(hours=cut)
    }
    assert full_rows.keys() == modified_rows.keys()
    for timestamp, row in full_rows.items():
        for name in FEATURE_COLUMNS:
            assert row[name] == modified_rows[timestamp][name]


def test_statistics_cover_features_and_target():
    build = build_training_dataset(make_source())

    stats = feature_statistics(build)

    assert set(stats) == {*build.feature_columns, build.target_column}
    assert all(entry["null_count"] == 0 for entry in stats.values())
    assert stats[TARGET_COLUMN]["min"] == pytest.approx(5.0 + 72 * 0.1, abs=1e-5)


def test_missing_raw_feature_column_is_rejected():
    source = make_source().drop_columns(["wind_capacity_mw"])

    with pytest.raises(ValueError, match="wind_capacity_mw"):
        build_training_dataset(source)


def test_missing_target_column_is_rejected():
    source = make_source().drop_columns([SOURCE_TARGET_COLUMN])

    with pytest.raises(ValueError, match=SOURCE_TARGET_COLUMN):
        build_training_dataset(source)


def test_all_incomplete_rows_raise():
    source = make_source(rows=80)
    # Only 80 rows; the first 72 have null rolling windows and the last has no
    # target, so at most seven rows survive. Remove their spot prices instead:
    values = source[SOURCE_TARGET_COLUMN].to_pylist()
    source = source.set_column(
        source.schema.get_field_index(SOURCE_TARGET_COLUMN),
        SOURCE_TARGET_COLUMN,
        pa.array([values[0]] + [None] * (len(values) - 1), type=pa.float32()),
    )

    with pytest.raises(RuntimeError, match="Every dataset row"):
        build_training_dataset(source)