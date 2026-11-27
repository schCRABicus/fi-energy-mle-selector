from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from fi_energy.features.spec import FeatureSpec


def make_spec(**overrides) -> FeatureSpec:
    values = {
        "name": "wind_penetration_ratio",
        "inputs": ("wind_mw", "consumption_mw"),
        "group": "grid_ratios",
        "description": "Share of consumption met by wind.",
    }
    values.update(overrides)
    return FeatureSpec(**values)


def test_feature_spec_stores_metadata():
    spec = make_spec()

    assert spec.name == "wind_penetration_ratio"
    assert spec.inputs == ("wind_mw", "consumption_mw")
    assert spec.group == "grid_ratios"
    assert spec.description == "Share of consumption met by wind."


def test_feature_spec_normalizes_inputs_to_a_tuple():
    spec = make_spec(inputs=["wind_mw", "consumption_mw"])

    assert spec.inputs == ("wind_mw", "consumption_mw")
    assert isinstance(spec.inputs, tuple)


def test_feature_spec_is_immutable():
    spec = make_spec()

    with pytest.raises(FrozenInstanceError):
        spec.name = "other"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"name": ""}, "must not be empty"),
        ({"name": "not-valid"}, "valid Python identifier"),
        ({"inputs": ()}, "at least one input"),
        ({"inputs": ("wind_mw", "wind_mw")}, "duplicate inputs"),
        ({"inputs": ("wind_penetration_ratio",)}, "itself as an input"),
        ({"inputs": "wind_mw"}, "not a string"),
        ({"group": "  "}, "non-empty group"),
        ({"description": ""}, "non-empty description"),
    ],
)
def test_feature_spec_rejects_invalid_metadata(overrides, message):
    with pytest.raises(ValueError, match=message):
        make_spec(**overrides)