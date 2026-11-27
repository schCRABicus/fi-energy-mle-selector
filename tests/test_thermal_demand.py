from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pyarrow as pa
import pytest

from fi_energy.features import (
    DEFAULT_CATALOG,
    FeatureCatalog,
    ThermalDemandFeatureGroup,
    build_feature_table,
    describe_features,
)
from fi_energy.features.thermal_demand import (
    HEATING_BASE_TEMPERATURE_C,
    HEATING_DEGREE_HOURS,
    TEMP_ROLLING_24H,
    TEMP_ROLLING_72H,
    TEMPERATURE_COLUMN,
    THERMAL_DEMAND_GROUP,
)

START = datetime(2024, 1, 1, tzinfo=UTC)
UTC_TIMESTAMP = pa.timestamp("us", tz="UTC")
FEATURE_COLUMNS = [HEATING_DEGREE_HOURS, TEMP_ROLLING_24H, TEMP_ROLLING_72H]
GROUP = ThermalDemandFeatureGroup()
THERMAL_ONLY_CATALOG = FeatureCatalog((GROUP,))


def make_table(
    temperatures: list[float | None],
    *,
    timestamps: list[datetime] | None = None,
) -> pa.Table:
    schema = pa.schema([
        pa.field("timestamp_utc", UTC_TIMESTAMP, nullable=False),
        pa.field(TEMPERATURE_COLUMN, pa.float32()),
    ])
    rows = len(temperatures)
    return pa.Table.from_pydict(
        {
            "timestamp_utc": timestamps
            or [START + timedelta(hours=index) for index in range(rows)],
            TEMPERATURE_COLUMN: temperatures,
        },
        schema=schema,
    )


def make_full_table(temperatures: list[float | None]) -> pa.Table:
    rows = len(temperatures)
    schema = pa.schema([
        pa.field("timestamp_utc", UTC_TIMESTAMP, nullable=False),
        pa.field("wind_mw", pa.float32()),
        pa.field("consumption_mw", pa.float32()),
        pa.field("wind_capacity_mw", pa.float32()),
        pa.field(TEMPERATURE_COLUMN, pa.float32()),
    ])
    return pa.Table.from_pydict(
        {
            "timestamp_utc": [
                START + timedelta(hours=index) for index in range(rows)
            ],
            "wind_mw": [1000.0] * rows,
            "consumption_mw": [10000.0] * rows,
            "wind_capacity_mw": [8000.0] * rows,
            TEMPERATURE_COLUMN: temperatures,
        },
        schema=schema,
    )


def test_compute_appends_thermal_features_in_declaration_order():
    result = GROUP.compute(make_table([10.0, 17.0, 20.0]))

    assert result.column_names == [
        "timestamp_utc",
        TEMPERATURE_COLUMN,
        *FEATURE_COLUMNS,
    ]
    for name in FEATURE_COLUMNS:
        field = result.schema.field(name)
        assert field.type == pa.float32()
        assert field.nullable


def test_heating_degree_hours_uses_the_seventeen_degree_base():
    result = GROUP.compute(make_table([10.0, 17.0, 20.0, -5.0]))

    assert HEATING_BASE_TEMPERATURE_C == 17.0
    assert result[HEATING_DEGREE_HOURS].to_pylist() == [7.0, 0.0, 0.0, 22.0]


def test_heating_degree_hours_propagates_null_temperatures():
    result = GROUP.compute(make_table([None, 10.0]))

    assert result[HEATING_DEGREE_HOURS].to_pylist() == [None, 7.0]


def test_rolling_temperature_looks_only_backwards():
    # 71 mild hours followed by one cold hour: only the last row has a full
    # 72-hour window, and its 24-hour mean must reflect the cold hour.
    temperatures = [10.0] * 71 + [-2.0]

    result = GROUP.compute(make_table(temperatures))

    rolling_24h = result[TEMP_ROLLING_24H].to_pylist()
    rolling_72h = result[TEMP_ROLLING_72H].to_pylist()
    assert rolling_24h[22] is None
    assert rolling_24h[23] == pytest.approx(10.0)
    assert rolling_24h[70] == pytest.approx(10.0)
    assert rolling_24h[71] == pytest.approx((23 * 10.0 - 2.0) / 24)
    assert rolling_72h[70] is None
    assert rolling_72h[71] == pytest.approx((71 * 10.0 - 2.0) / 72)


def test_rolling_temperature_skips_missing_readings():
    temperatures = [10.0] * 71 + [None, 4.0]

    result = GROUP.compute(make_table(temperatures))

    # The final window has 22 mild hours plus the 4 °C reading; the null is
    # skipped rather than turning the whole window null.
    assert result[TEMP_ROLLING_24H].to_pylist()[72] == pytest.approx(
        (22 * 10.0 + 4.0) / 23
    )


def test_compute_preserves_source_columns_and_metadata():
    table = make_table([10.0, 11.0]).replace_schema_metadata({b"k": b"v"})
    original_columns = table.column_names

    result = GROUP.compute(table)

    assert result.column_names[: len(original_columns)] == original_columns
    assert result.schema.metadata == {b"k": b"v"}
    assert table.column_names == original_columns


def test_compute_rejects_missing_input():
    table = pa.table({"timestamp_utc": [START]})

    with pytest.raises(ValueError, match="missing required column"):
        GROUP.compute(table)


def test_compute_rejects_non_numeric_input():
    table = pa.table({"timestamp_utc": [START], TEMPERATURE_COLUMN: ["cold"]})

    with pytest.raises(ValueError, match="must be numeric"):
        GROUP.compute(table)


def test_compute_rejects_non_table():
    with pytest.raises(TypeError, match="pyarrow Table"):
        GROUP.compute({})  # type: ignore[arg-type]


def test_compute_rejects_existing_feature_columns():
    table = make_table([10.0]).append_column(
        HEATING_DEGREE_HOURS,
        pa.array([7.0], type=pa.float32()),
    )

    with pytest.raises(ValueError, match="already has feature column"):
        GROUP.compute(table)


def test_compute_handles_an_empty_table():
    result = GROUP.compute(make_table([]))

    assert result.num_rows == 0
    for name in FEATURE_COLUMNS:
        assert result.schema.field(name).type == pa.float32()


def test_diagnostics_summarise_temperature_and_heating_demand():
    table = make_table([10.0, 17.0, 20.0, None, -5.0])

    diagnostics = GROUP.diagnostics(table)

    assert diagnostics["temperature_nulls"] == 1
    assert diagnostics["heating_hours"] == 2
    assert diagnostics["max_heating_degree_hours"] == pytest.approx(22.0)


def test_diagnostics_handle_an_all_null_temperature_column():
    diagnostics = GROUP.diagnostics(make_table([None, None]))

    assert diagnostics == {
        "temperature_nulls": 2,
        "heating_hours": 0,
        "max_heating_degree_hours": None,
    }


def test_build_feature_table_with_the_thermal_group():
    table = make_table([10.0] * 72)

    result = build_feature_table(table, catalog=THERMAL_ONLY_CATALOG)
    description = describe_features(
        table,
        result,
        catalog=THERMAL_ONLY_CATALOG,
    )

    assert result.column_names == ["timestamp_utc", *FEATURE_COLUMNS]
    assert description["groups"] == [THERMAL_DEMAND_GROUP]
    assert description["feature_columns"] == FEATURE_COLUMNS
    assert [entry["group"] for entry in description["features"]] == [
        THERMAL_DEMAND_GROUP,
        THERMAL_DEMAND_GROUP,
        THERMAL_DEMAND_GROUP,
    ]
    assert description["diagnostics"][THERMAL_DEMAND_GROUP] == {
        "temperature_nulls": 0,
        "heating_hours": 72,
        "max_heating_degree_hours": pytest.approx(7.0),
    }
    assert description["null_values"] == {
        HEATING_DEGREE_HOURS: 0,
        TEMP_ROLLING_24H: 23,
        TEMP_ROLLING_72H: 71,
    }


def test_build_feature_table_with_the_default_catalog():
    table = make_full_table([10.0] * 72)

    result = build_feature_table(table)

    assert result.column_names == [
        "timestamp_utc",
        *DEFAULT_CATALOG.feature_names,
    ]
    assert result.num_rows == 72