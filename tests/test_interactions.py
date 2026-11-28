from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pyarrow as pa
import pytest

from fi_energy.features import (
    DEFAULT_CATALOG,
    FeatureCatalog,
    InteractionFeatureGroup,
    build_feature_table,
    describe_features,
)
from fi_energy.features.groups.interactions import (
    INTERACTIONS_GROUP,
    WIND_TEMP_INTERACTION,
)
from fi_energy.features.groups.thermal_demand import HEATING_DEGREE_HOURS
from fi_energy.features.groups.wind_ratios import WIND_PENETRATION_RATIO

START = datetime(2024, 1, 1, tzinfo=UTC)
UTC_TIMESTAMP = pa.timestamp("us", tz="UTC")
GROUP = InteractionFeatureGroup()


def make_table(
    penetration: list[float | None],
    degree_hours: list[float | None],
) -> pa.Table:
    return pa.table(
        {
            "timestamp_utc": pa.array(
                [
                    START + timedelta(hours=index)
                    for index in range(len(penetration))
                ],
                type=UTC_TIMESTAMP,
            ),
            WIND_PENETRATION_RATIO: pa.array(penetration, type=pa.float32()),
            HEATING_DEGREE_HOURS: pa.array(degree_hours, type=pa.float32()),
        }
    )


def make_source(rows: int = 80) -> pa.Table:
    schema = pa.schema([
        pa.field("timestamp_utc", UTC_TIMESTAMP, nullable=False),
        pa.field("wind_mw", pa.float32()),
        pa.field("consumption_mw", pa.float32()),
        pa.field("wind_capacity_mw", pa.float32()),
        pa.field("helsinki_temperature_c", pa.float32()),
    ])
    return pa.Table.from_pydict(
        {
            "timestamp_utc": [
                START + timedelta(hours=index) for index in range(rows)
            ],
            "wind_mw": [1000.0] * rows,
            "consumption_mw": [10000.0] * rows,
            "wind_capacity_mw": [8000.0] * rows,
            "helsinki_temperature_c": [
                10.0 if index % 2 else 0.0 for index in range(rows)
            ],
        },
        schema=schema,
    )


def test_compute_multiplies_wind_share_by_heating_degree_hours():
    result = GROUP.compute(
        make_table([0.25, 0.0, 0.5], [7.0, 12.0, 0.0])
    )

    assert result.column_names == [
        "timestamp_utc",
        WIND_PENETRATION_RATIO,
        HEATING_DEGREE_HOURS,
        WIND_TEMP_INTERACTION,
    ]
    assert result[WIND_TEMP_INTERACTION].to_pylist() == [1.75, 0.0, 0.0]
    field = result.schema.field(WIND_TEMP_INTERACTION)
    assert field.type == pa.float32()
    assert field.nullable


def test_compute_propagates_nulls_from_either_input():
    result = GROUP.compute(
        make_table([None, 0.5, 0.5], [10.0, None, 10.0])
    )

    assert result[WIND_TEMP_INTERACTION].to_pylist() == [None, None, 5.0]


def test_compute_preserves_source_columns_and_metadata():
    table = make_table([0.25], [7.0]).replace_schema_metadata({b"k": b"v"})

    result = GROUP.compute(table)

    assert result.column_names[:3] == table.column_names
    assert result.schema.metadata == {b"k": b"v"}


def test_compute_rejects_missing_input():
    table = pa.table({"timestamp_utc": [START], WIND_PENETRATION_RATIO: [0.2]})

    with pytest.raises(ValueError, match="missing required column"):
        GROUP.compute(table)


def test_compute_rejects_non_numeric_input():
    table = pa.table({
        "timestamp_utc": [START],
        WIND_PENETRATION_RATIO: ["0.2"],
        HEATING_DEGREE_HOURS: [7.0],
    })

    with pytest.raises(ValueError, match="must be numeric"):
        GROUP.compute(table)


def test_compute_rejects_non_table():
    with pytest.raises(TypeError, match="pyarrow Table"):
        GROUP.compute({})  # type: ignore[arg-type]


def test_compute_rejects_an_existing_feature_column():
    table = make_table([0.2], [7.0]).append_column(
        WIND_TEMP_INTERACTION,
        pa.array([1.4], type=pa.float32()),
    )

    with pytest.raises(ValueError, match="already has feature column"):
        GROUP.compute(table)


def test_compute_handles_an_empty_table():
    result = GROUP.compute(make_table([], []))

    assert result.num_rows == 0
    assert result.schema.field(WIND_TEMP_INTERACTION).type == pa.float32()


def test_catalog_requires_the_producers_before_the_interaction():
    with pytest.raises(ValueError, match="before it is produced"):
        FeatureCatalog((
            InteractionFeatureGroup(),
            *(
                group
                for group in DEFAULT_CATALOG.groups
                if group.name != INTERACTIONS_GROUP
            ),
        ))


def test_subset_cannot_drop_the_interaction_inputs():
    with pytest.raises(ValueError, match="before it is produced"):
        DEFAULT_CATALOG.subset(INTERACTIONS_GROUP)


def test_default_catalog_builds_the_interaction_from_earlier_features():
    source = make_source()

    result = build_feature_table(source)
    description = describe_features(source, result)

    assert WIND_TEMP_INTERACTION in result.column_names
    assert result.num_rows == source.num_rows
    penetration = result[WIND_PENETRATION_RATIO].to_pylist()
    degree_hours = result[HEATING_DEGREE_HOURS].to_pylist()
    interaction = result[WIND_TEMP_INTERACTION].to_pylist()
    for index in (0, 30, 79):
        assert interaction[index] == pytest.approx(
            penetration[index] * degree_hours[index],
            rel=1e-6,
        )

    entry = next(
        entry
        for entry in description["features"]
        if entry["name"] == WIND_TEMP_INTERACTION
    )
    assert entry["group"] == INTERACTIONS_GROUP
    assert entry["inputs"] == (
        WIND_PENETRATION_RATIO,
        HEATING_DEGREE_HOURS,
    )
    assert INTERACTIONS_GROUP in description["groups"]