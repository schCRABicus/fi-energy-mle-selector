from __future__ import annotations

import pyarrow as pa
import pytest

from fi_energy.features import FeatureGroup, FeatureSpec


def make_spec(
    name: str = "dummy_feature",
    inputs: tuple[str, ...] = ("x",),
) -> FeatureSpec:
    return FeatureSpec(name=name, inputs=inputs, description="test")


def make_table() -> pa.Table:
    return pa.table({"x": pa.array([1.0, 2.0], type=pa.float64())})


class DummyGroup(FeatureGroup):
    name = "dummy"
    specs = (make_spec(),)

    def _compute(self, table: pa.Table) -> pa.Table:
        return table.append_column(
            "dummy_feature",
            pa.array([1.0] * table.num_rows, type=pa.float32()),
        )


def test_compute_appends_declared_feature_columns():
    result = DummyGroup().compute(make_table())

    assert result.column_names == ["x", "dummy_feature"]
    assert result.schema.field("dummy_feature").type == pa.float32()
    assert result["dummy_feature"].to_pylist() == [1.0, 1.0]


def test_compute_rejects_missing_input():
    with pytest.raises(ValueError, match="missing required column"):
        DummyGroup().compute(pa.table({"y": [1.0]}))


def test_compute_rejects_existing_feature_column():
    table = make_table().append_column(
        "dummy_feature",
        pa.array([0.5, 0.5], type=pa.float32()),
    )

    with pytest.raises(ValueError, match="already has feature column"):
        DummyGroup().compute(table)


def test_compute_rejects_non_table():
    with pytest.raises(TypeError, match="pyarrow Table"):
        DummyGroup().compute({})  # type: ignore[arg-type]


def test_compute_validates_before_running():
    calls: list[bool] = []

    class SpyGroup(FeatureGroup):
        name = "spy"
        specs = (FeatureSpec("spy_feature", ("missing",), "test"),)

        def _compute(self, table: pa.Table) -> pa.Table:
            calls.append(True)
            return table

    with pytest.raises(ValueError, match="missing required column"):
        SpyGroup().compute(make_table())

    assert calls == []


def test_feature_names_and_inputs_are_derived_from_specs():
    class TwoFeatureGroup(FeatureGroup):
        name = "two"
        specs = (
            FeatureSpec("first", ("x", "y"), "test"),
            FeatureSpec("second", ("x", "z"), "test"),
        )

        def _compute(self, table: pa.Table) -> pa.Table:
            for name in self.feature_names:
                table = table.append_column(
                    name,
                    pa.array([0.0] * table.num_rows, type=pa.float32()),
                )
            return table

    group = TwoFeatureGroup()

    assert group.feature_names == ("first", "second")
    assert group.inputs == ("x", "y", "z")


def test_diagnostics_defaults_to_empty():
    assert DummyGroup().diagnostics(make_table()) == {}


@pytest.mark.parametrize(
    ("attrs", "match"),
    [
        ({"specs": (make_spec(),)}, "non-empty string 'name'"),
        ({"name": "", "specs": (make_spec(),)}, "non-empty string 'name'"),
        ({"name": "bad name", "specs": (make_spec(),)}, "identifier"),
        ({"name": "dummy"}, "non-empty tuple 'specs'"),
        ({"name": "dummy", "specs": ()}, "non-empty tuple 'specs'"),
        (
            {"name": "dummy", "specs": [make_spec()]},
            "non-empty tuple 'specs'",
        ),
        ({"name": "dummy", "specs": ("nope",)}, "FeatureSpec values"),
        (
            {"name": "dummy", "specs": (make_spec(), make_spec())},
            "duplicate feature",
        ),
    ],
)
def test_group_declaration_validation(
    attrs: dict[str, object],
    match: str,
) -> None:
    with pytest.raises(TypeError, match=match):
        type(
            "BadGroup",
            (FeatureGroup,),
            {"_compute": lambda self, table: table, **attrs},
        )


def test_compute_rejects_missing_output_column():
    class MissingOutput(FeatureGroup):
        name = "missing_output"
        specs = (FeatureSpec("out", ("x",), "test"),)

        def _compute(self, table: pa.Table) -> pa.Table:
            return table

    with pytest.raises(RuntimeError, match="must append exactly"):
        MissingOutput().compute(make_table())


def test_compute_rejects_extra_output_column():
    class ExtraOutput(FeatureGroup):
        name = "extra_output"
        specs = (FeatureSpec("out", ("x",), "test"),)

        def _compute(self, table: pa.Table) -> pa.Table:
            table = table.append_column(
                "other",
                pa.array([0.0] * table.num_rows, type=pa.float32()),
            )
            return table.append_column(
                "out",
                pa.array([0.0] * table.num_rows, type=pa.float32()),
            )

    with pytest.raises(RuntimeError, match="must append exactly"):
        ExtraOutput().compute(make_table())


def test_compute_rejects_non_float32_output():
    class WrongDtype(FeatureGroup):
        name = "wrong_dtype"
        specs = (FeatureSpec("out", ("x",), "test"),)

        def _compute(self, table: pa.Table) -> pa.Table:
            return table.append_column(
                "out",
                pa.array([1] * table.num_rows, type=pa.int32()),
            )

    with pytest.raises(RuntimeError, match="nullable float32"):
        WrongDtype().compute(make_table())


def test_compute_rejects_non_nullable_output():
    class NonNullable(FeatureGroup):
        name = "non_nullable"
        specs = (FeatureSpec("out", ("x",), "test"),)

        def _compute(self, table: pa.Table) -> pa.Table:
            return table.append_column(
                pa.field("out", pa.float32(), nullable=False),
                pa.array([1.0] * table.num_rows, type=pa.float32()),
            )

    with pytest.raises(RuntimeError, match="nullable float32"):
        NonNullable().compute(make_table())


def test_compute_rejects_row_count_change():
    class DropsRows(FeatureGroup):
        name = "drops_rows"
        specs = (FeatureSpec("out", ("x",), "test"),)

        def _compute(self, table: pa.Table) -> pa.Table:
            return table.append_column(
                "out",
                pa.array([1.0] * table.num_rows, type=pa.float32()),
            ).slice(0, 1)

    with pytest.raises(RuntimeError, match="changed the row count"):
        DropsRows().compute(make_table())