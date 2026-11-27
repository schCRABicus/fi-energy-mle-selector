from __future__ import annotations

import json
import math
from datetime import UTC, datetime, timedelta

import pyarrow as pa
import pytest

from fi_energy.features.build import ALL_FEATURES, build_feature_table, describe_features
from fi_energy.features.grid import (
    GRID_RATIO_FEATURES,
    GRID_RATIO_GROUP,
    add_grid_ratio_features,
    grid_ratio_diagnostics,
)
from fi_energy.features.spec import FeatureSpec

START = datetime(2024, 1, 1, tzinfo=UTC)
UTC_TIMESTAMP = pa.timestamp("us", tz="UTC")
FEATURE_COLUMNS = ["wind_penetration_ratio", "wind_capacity_factor"]


def make_table(
    winds: list[float | None],
    consumptions: list[float | None],
    capacities: list[float | None],
    *,
    timestamps: list[datetime] | None = None,
) -> pa.Table:
    schema = pa.schema([
        pa.field("timestamp_utc", UTC_TIMESTAMP, nullable=False),
        pa.field("wind_mw", pa.float32()),
        pa.field("consumption_mw", pa.float32()),
        pa.field("wind_capacity_mw", pa.float32()),
    ])
    rows = len(winds)
    return pa.Table.from_pydict(
        {
            "timestamp_utc": timestamps
            or [START + timedelta(hours=index) for index in range(rows)],
            "wind_mw": winds,
            "consumption_mw": consumptions,
            "wind_capacity_mw": capacities,
        },
        schema=schema,
    )


def test_add_grid_ratio_features_appends_ratios_in_registry_order():
    table = make_table([2500.0], [10000.0], [8000.0])

    result = add_grid_ratio_features(table)

    assert result.column_names == [*table.column_names, *FEATURE_COLUMNS]
    assert result["wind_penetration_ratio"].to_pylist() == [
        pytest.approx(0.25)
    ]
    assert result["wind_capacity_factor"].to_pylist() == [
        pytest.approx(0.3125)
    ]


def test_add_grid_ratio_features_preserves_source_columns_and_metadata():
    table = make_table([2500.0], [10000.0], [8000.0]).replace_schema_metadata({
        b"k": b"v"
    })
    original_columns = table.column_names

    result = add_grid_ratio_features(table)

    assert result.column_names[: len(original_columns)] == original_columns
    assert result["timestamp_utc"].to_pylist() == table["timestamp_utc"].to_pylist()
    assert result.schema.field("timestamp_utc").type == UTC_TIMESTAMP
    assert not result.schema.field("timestamp_utc").nullable
    assert result.schema.metadata == {b"k": b"v"}
    assert table.column_names == original_columns


def test_add_grid_ratio_features_propagates_nulls_per_input():
    table = make_table(
        [None, 1000.0, 1000.0],
        [10000.0, None, 10000.0],
        [8000.0, 8000.0, None],
    )

    result = add_grid_ratio_features(table)

    assert result["wind_penetration_ratio"].to_pylist() == [
        None,
        None,
        pytest.approx(0.1),
    ]
    assert result["wind_capacity_factor"].to_pylist() == [
        None,
        pytest.approx(0.125),
        None,
    ]


def test_add_grid_ratio_features_nulls_zero_denominators():
    table = make_table([100.0, 100.0], [0.0, 10000.0], [0.0, 8000.0])

    result = add_grid_ratio_features(table)

    penetration = result["wind_penetration_ratio"].to_pylist()
    capacity_factor = result["wind_capacity_factor"].to_pylist()
    assert penetration[0] is None
    assert capacity_factor[0] is None
    assert penetration[1] == pytest.approx(0.01)
    assert capacity_factor[1] == pytest.approx(0.0125)
    assert all(
        value is None or not math.isinf(value)
        for value in [*penetration, *capacity_factor]
    )


def test_add_grid_ratio_features_clips_negative_wind_to_zero():
    table = make_table(
        [-32.4, -1.0, 15000.0],
        [10000.0, 10000.0, 10000.0],
        [8000.0, 8000.0, 8000.0],
    )

    result = add_grid_ratio_features(table)

    assert result["wind_penetration_ratio"].to_pylist()[:2] == [0.0, 0.0]
    assert result["wind_capacity_factor"].to_pylist()[:2] == [0.0, 0.0]
    assert result["wind_penetration_ratio"].to_pylist()[2] == pytest.approx(1.5)
    assert result["wind_capacity_factor"].to_pylist()[2] == pytest.approx(1.875)
    assert result["wind_mw"].to_pylist()[0] == pytest.approx(-32.4)


def test_add_grid_ratio_features_keeps_null_wind_while_clipping_neighbours():
    table = make_table([None, -5.0], [10000.0, 10000.0], [8000.0, 8000.0])

    result = add_grid_ratio_features(table)

    assert result["wind_penetration_ratio"].to_pylist() == [None, 0.0]
    assert result["wind_capacity_factor"].to_pylist() == [None, 0.0]


def test_add_grid_ratio_features_rejects_missing_input():
    table = pa.table({
        "timestamp_utc": [START],
        "wind_mw": [1.0],
        "consumption_mw": [2.0],
    })

    with pytest.raises(ValueError, match="missing required column"):
        add_grid_ratio_features(table)


def test_add_grid_ratio_features_rejects_non_numeric_input():
    table = pa.table({
        "timestamp_utc": [START],
        "wind_mw": ["1.0"],
        "consumption_mw": [2.0],
        "wind_capacity_mw": [3.0],
    })

    with pytest.raises(ValueError, match="must be numeric"):
        add_grid_ratio_features(table)


def test_add_grid_ratio_features_rejects_non_table():
    with pytest.raises(TypeError, match="pyarrow Table"):
        add_grid_ratio_features({})  # type: ignore[arg-type]


def test_add_grid_ratio_features_rejects_existing_feature_columns():
    table = make_table([1.0], [2.0], [3.0]).append_column(
        "wind_penetration_ratio",
        pa.array([0.5], type=pa.float32()),
    )

    with pytest.raises(ValueError, match="already has feature column"):
        add_grid_ratio_features(table)


def test_add_grid_ratio_features_handles_an_empty_table():
    table = make_table([], [], [])

    result = add_grid_ratio_features(table)

    assert result.num_rows == 0
    assert result.schema.field("wind_penetration_ratio").type == pa.float32()
    assert result.schema.field("wind_capacity_factor").type == pa.float32()
    assert result["wind_penetration_ratio"].to_pylist() == []


def test_grid_ratio_registry_metadata():
    assert ALL_FEATURES == GRID_RATIO_FEATURES
    assert [spec.name for spec in GRID_RATIO_FEATURES] == FEATURE_COLUMNS
    assert all(isinstance(spec, FeatureSpec) for spec in GRID_RATIO_FEATURES)
    assert all(
        spec.group == GRID_RATIO_GROUP for spec in GRID_RATIO_FEATURES
    )
    assert GRID_RATIO_FEATURES[0].inputs == ("wind_mw", "consumption_mw")
    assert GRID_RATIO_FEATURES[1].inputs == ("wind_mw", "wind_capacity_mw")
    assert all(spec.description.strip() for spec in GRID_RATIO_FEATURES)


def test_build_feature_table_selects_timestamp_and_features():
    table = make_table(
        [2500.0, None],
        [10000.0, 10000.0],
        [8000.0, 8000.0],
    )

    result = build_feature_table(table)

    assert result.column_names == ["timestamp_utc", *FEATURE_COLUMNS]
    assert result.num_rows == table.num_rows
    assert result["timestamp_utc"].to_pylist() == table["timestamp_utc"].to_pylist()


def test_build_feature_table_preserves_source_metadata():
    table = make_table([2500.0], [10000.0], [8000.0]).replace_schema_metadata({
        b"k": b"v"
    })

    result = build_feature_table(table)

    assert result.schema.metadata == {b"k": b"v"}


def test_build_feature_table_rejects_empty_source():
    with pytest.raises(ValueError, match="no rows"):
        build_feature_table(make_table([], [], []))


def test_build_feature_table_rejects_missing_timestamp():
    table = pa.table({
        "wind_mw": [1.0],
        "consumption_mw": [2.0],
        "wind_capacity_mw": [3.0],
    })

    with pytest.raises(ValueError, match="no 'timestamp_utc'"):
        build_feature_table(table)


def test_build_feature_table_rejects_non_utc_timestamp_type():
    table = make_table([1.0], [2.0], [3.0]).set_column(
        0,
        "timestamp_utc",
        pa.array([START], type=pa.timestamp("us", tz="Europe/Helsinki")),
    )

    with pytest.raises(ValueError, match="must be"):
        build_feature_table(table)


def test_build_feature_table_rejects_null_timestamps():
    table = make_table([1.0], [2.0], [3.0]).set_column(
        0,
        "timestamp_utc",
        pa.array([None], type=UTC_TIMESTAMP),
    )

    with pytest.raises(ValueError, match="null values"):
        build_feature_table(table)


def test_build_feature_table_rejects_duplicate_timestamps():
    table = make_table(
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
        timestamps=[START, START],
    )

    with pytest.raises(ValueError, match="duplicate timestamp"):
        build_feature_table(table)


def test_build_feature_table_rejects_off_grid_timestamps():
    table = make_table(
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
        timestamps=[
            START,
            START + timedelta(minutes=30),
            START + timedelta(hours=1),
        ],
    )

    with pytest.raises(ValueError, match="hourly grid"):
        build_feature_table(table)


def test_build_feature_table_rejects_non_table():
    with pytest.raises(TypeError, match="pyarrow Table"):
        build_feature_table({})  # type: ignore[arg-type]


def test_describe_features_reports_policy_and_statistics():
    table = make_table(
        [-5.0, 1000.0, 2000.0],
        [0.0, 10000.0, 10000.0],
        [8000.0, 0.0, 8000.0],
    )
    features = build_feature_table(table)

    description = describe_features(table, features)

    assert description["rows"] == 3
    assert description["feature_columns"] == FEATURE_COLUMNS
    assert description["diagnostics"]["negative_wind_count"] == 1
    assert description["diagnostics"]["negative_wind_min_mw"] == pytest.approx(
        -5.0
    )
    assert description["diagnostics"]["zero_denominators"] == {
        "wind_penetration_ratio": 1,
        "wind_capacity_factor": 1,
    }
    assert description["null_values"] == {
        "wind_penetration_ratio": 1,
        "wind_capacity_factor": 1,
    }
    assert description["input_null_counts"] == {
        "wind_mw": 0,
        "consumption_mw": 0,
        "wind_capacity_mw": 0,
    }
    assert description["statistics"]["wind_penetration_ratio"]["min"] == (
        pytest.approx(0.1)
    )
    assert description["statistics"]["wind_penetration_ratio"]["max"] == (
        pytest.approx(0.2)
    )
    assert description["statistics"]["wind_penetration_ratio"]["mean"] == (
        pytest.approx(0.15)
    )
    assert description["statistics"]["wind_capacity_factor"]["min"] == (
        pytest.approx(0.0)
    )
    assert description["statistics"]["wind_capacity_factor"]["max"] == (
        pytest.approx(0.25)
    )
    assert description["statistics"]["wind_capacity_factor"]["mean"] == (
        pytest.approx(0.125)
    )
    json.dumps(description)


def test_grid_ratio_diagnostics_summarises_the_raw_inputs():
    table = make_table(
        [-32.4, -1.0, 1000.0],
        [0.0, 10000.0, 10000.0],
        [0.0, 0.0, 8000.0],
    )

    diagnostics = grid_ratio_diagnostics(table)

    assert diagnostics["negative_wind_count"] == 2
    assert diagnostics["negative_wind_min_mw"] == pytest.approx(-32.4)
    assert diagnostics["zero_denominators"] == {
        "wind_penetration_ratio": 1,
        "wind_capacity_factor": 2,
    }