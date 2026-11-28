from __future__ import annotations

from typing import Any

import pyarrow as pa
import pytest

from fi_energy.features import (
    DEFAULT_CATALOG,
    FeatureCatalog,
    FeatureGroup,
    FeatureSpec,
)


class AddA(FeatureGroup):
    name = "add_a"
    specs = (FeatureSpec("a_feature", ("x",), "test"),)

    def _compute(self, table: pa.Table) -> pa.Table:
        return table.append_column(
            "a_feature",
            pa.array([1.0] * table.num_rows, type=pa.float32()),
        )

    def diagnostics(self, table: pa.Table) -> dict[str, Any]:
        return {"rows": table.num_rows}


class AddB(FeatureGroup):
    name = "add_b"
    specs = (FeatureSpec("b_feature", ("a_feature",), "test"),)

    def _compute(self, table: pa.Table) -> pa.Table:
        values = table["a_feature"].cast(pa.float32())
        return table.append_column(
            "b_feature",
            pa.compute.multiply(values, pa.scalar(2.0, pa.float32())),
        )


class DuplicateOfAddA(FeatureGroup):
    name = "add_a"
    specs = (FeatureSpec("other_feature", ("x",), "test"),)

    def _compute(self, table: pa.Table) -> pa.Table:
        return table


class AlsoProducesA(FeatureGroup):
    name = "also_a"
    specs = (FeatureSpec("a_feature", ("x",), "test"),)

    def _compute(self, table: pa.Table) -> pa.Table:
        return table


def make_table() -> pa.Table:
    return pa.table({"x": pa.array([1.0, 2.0], type=pa.float32())})


def test_catalog_derives_specs_and_feature_names_in_order():
    catalog = FeatureCatalog((AddA(), AddB()))

    assert catalog.feature_names == ("a_feature", "b_feature")
    assert [spec.name for spec in catalog.specs] == ["a_feature", "b_feature"]


def test_catalog_normalizes_groups_to_a_tuple():
    catalog = FeatureCatalog([AddA()])  # type: ignore[arg-type]

    assert isinstance(catalog.groups, tuple)


def test_catalog_is_immutable():
    catalog = FeatureCatalog((AddA(),))

    with pytest.raises(AttributeError):
        catalog.groups = ()  # type: ignore[misc]


def test_catalog_rejects_an_empty_group_list():
    with pytest.raises(ValueError, match="at least one group"):
        FeatureCatalog(())


def test_catalog_rejects_non_group_values():
    with pytest.raises(TypeError, match="FeatureGroup values"):
        FeatureCatalog(("not a group",))  # type: ignore[arg-type]


def test_catalog_rejects_duplicate_group_names():
    with pytest.raises(ValueError, match="Duplicate feature group name"):
        FeatureCatalog((AddA(), DuplicateOfAddA()))


def test_catalog_rejects_duplicate_feature_names():
    with pytest.raises(ValueError, match="produced by more than one group"):
        FeatureCatalog((AddA(), AlsoProducesA()))


def test_catalog_rejects_out_of_order_dependencies():
    with pytest.raises(ValueError, match="before it is produced"):
        FeatureCatalog((AddB(), AddA()))


def test_group_lookup():
    group = AddA()
    catalog = FeatureCatalog((group, AddB()))

    assert catalog.group("add_a") is group
    with pytest.raises(KeyError, match="no_group"):
        catalog.group("no_group")


def test_subset_keeps_catalog_order():
    catalog = FeatureCatalog((AddA(), AddB()))

    subset = catalog.subset("add_b", "add_a")

    assert subset.feature_names == ("a_feature", "b_feature")


def test_subset_rejects_an_empty_selection():
    with pytest.raises(ValueError, match="at least one feature group"):
        DEFAULT_CATALOG.subset()


def test_subset_rejects_unknown_groups():
    with pytest.raises(KeyError, match="Unknown feature group"):
        DEFAULT_CATALOG.subset("missing")


def test_subset_rejects_a_break_in_dependency_order():
    catalog = FeatureCatalog((AddA(), AddB()))

    with pytest.raises(ValueError, match="before it is produced"):
        catalog.subset("add_b")


def test_compute_applies_groups_in_order():
    catalog = FeatureCatalog((AddA(), AddB()))

    result = catalog.compute(make_table())

    assert result.column_names == ["x", "a_feature", "b_feature"]
    assert result["b_feature"].to_pylist() == [2.0, 2.0]


def test_diagnostics_are_keyed_by_group_name():
    catalog = FeatureCatalog((AddA(), AddB()))

    assert catalog.diagnostics(make_table()) == {
        "add_a": {"rows": 2},
        "add_b": {},
    }


def test_default_catalog_contains_the_registered_groups():
    assert [group.name for group in DEFAULT_CATALOG.groups] == [
        "wind_ratios",
        "thermal_demand",
        "interactions",
        "calendar",
    ]
    assert DEFAULT_CATALOG.feature_names == (
        "wind_penetration_ratio",
        "wind_capacity_factor",
        "heating_degree_hours",
        "temp_rolling_24h",
        "temp_rolling_72h",
        "wind_temp_interaction",
        "sin_hour",
        "cos_hour",
        "sin_month",
        "cos_month",
        "is_weekend",
        "is_fi_public_holiday",
    )