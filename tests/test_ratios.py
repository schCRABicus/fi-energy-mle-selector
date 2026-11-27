from __future__ import annotations

import math

import pyarrow as pa
import pytest

from fi_energy.features.ratios import safe_ratio

FLOAT32 = pa.float32()


def float32(values: list[float | None]) -> pa.Array:
    return pa.array(values, type=FLOAT32)


def test_safe_ratio_computes_float32_quotients():
    result = safe_ratio(
        float32([1.0, 3.0, None]),
        float32([2.0, 4.0, 2.0]),
    )

    assert result.type.equals(FLOAT32)
    assert result.to_pylist() == [0.5, 0.75, None]


def test_safe_ratio_promotes_integer_inputs():
    result = safe_ratio(
        pa.array([1, 3], type=pa.int64()),
        pa.array([2, 2], type=pa.int64()),
    )

    assert result.type.equals(FLOAT32)
    assert result.to_pylist() == [0.5, 1.5]


@pytest.mark.parametrize("denominator", [0.0, -0.0])
def test_safe_ratio_turns_zero_denominators_into_null(denominator):
    result = safe_ratio(
        float32([1.0, 2.0]),
        float32([denominator, 4.0]),
    )

    values = result.to_pylist()
    assert values[0] is None
    assert values[1] == 0.5
    assert all(value is None or not math.isinf(value) for value in values)


def test_safe_ratio_propagates_null_denominators():
    result = safe_ratio(float32([1.0]), float32([None]))

    assert result.to_pylist() == [None]


def test_safe_ratio_accepts_chunked_arrays():
    numerator = pa.chunked_array([[1.0, 2.0], [3.0]], type=FLOAT32)
    denominator = pa.chunked_array([[2.0], [2.0, 4.0]], type=FLOAT32)

    result = safe_ratio(numerator, denominator)

    assert isinstance(result, pa.ChunkedArray)
    assert result.to_pylist() == [0.5, 1.0, 0.75]


def test_safe_ratio_rejects_length_mismatch():
    with pytest.raises(ValueError, match="same length"):
        safe_ratio(float32([1.0]), float32([1.0, 2.0]))


def test_safe_ratio_rejects_non_numeric_inputs():
    with pytest.raises(ValueError, match="must be numeric"):
        safe_ratio(pa.array(["1"]), float32([2.0]))


def test_safe_ratio_rejects_non_arrow_inputs():
    with pytest.raises(TypeError, match="pyarrow"):
        safe_ratio([1.0], [2.0])  # type: ignore[arg-type]