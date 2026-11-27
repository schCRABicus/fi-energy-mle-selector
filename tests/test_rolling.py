from __future__ import annotations

import pyarrow as pa
import pytest

from fi_energy.features.rolling import trailing_mean


def test_trailing_mean_uses_the_preceding_window():
    result = trailing_mean(pa.array([1.0, 2.0, 3.0, 4.0]), 2)

    assert result.to_pylist() == [None, 1.5, 2.5, 3.5]
    assert result.type == pa.float32()


def test_trailing_mean_requires_a_full_window():
    result = trailing_mean(pa.array([1.0, 2.0]), 3)

    assert result.to_pylist() == [None, None]
    assert len(result) == 2


def test_trailing_mean_skips_nulls_inside_the_window():
    result = trailing_mean(pa.array([1.0, None, 3.0]), 3)

    assert result.to_pylist() == [None, None, 2.0]


def test_trailing_mean_returns_null_when_a_window_has_no_observations():
    result = trailing_mean(pa.array([None, None, 3.0]), 3)

    assert result.to_pylist() == [None, None, 3.0]

    all_null = trailing_mean(
        pa.array([None, None, None], type=pa.float32()),
        2,
    )

    assert all_null.to_pylist() == [None, None, None]


def test_trailing_mean_never_uses_future_values():
    base = trailing_mean(pa.array([1.0, 2.0, 3.0, 4.0, 5.0]), 3)
    perturbed = trailing_mean(pa.array([1.0, 2.0, 3.0, 4.0, 100.0]), 3)

    assert base.to_pylist()[:-1] == perturbed.to_pylist()[:-1]
    assert base.to_pylist()[-1] != perturbed.to_pylist()[-1]


def test_trailing_mean_handles_empty_and_short_inputs():
    empty = trailing_mean(pa.array([], type=pa.float32()), 3)

    assert empty.to_pylist() == []
    assert empty.type == pa.float32()

    assert trailing_mean(pa.array([1.0]), 2).to_pylist() == [None]


def test_trailing_mean_accepts_chunked_and_integer_inputs():
    chunked = pa.chunked_array([[1.0, 2.0], [3.0]])

    assert trailing_mean(chunked, 2).to_pylist() == [None, 1.5, 2.5]

    integers = pa.array([0, 0, 100], type=pa.int32())

    assert trailing_mean(integers, 2).to_pylist() == [None, 0.0, 50.0]


def test_trailing_mean_matches_a_brute_force_window():
    values = [5.0, None, 7.0, 1.0, None, None, 9.0, 3.0, 2.0, 8.0]
    window = 4

    result = trailing_mean(pa.array(values), window).to_pylist()

    for index, actual in enumerate(result):
        if index + 1 < window:
            assert actual is None
            continue
        observed = [
            value
            for value in values[index + 1 - window : index + 1]
            if value is not None
        ]
        expected = sum(observed) / len(observed) if observed else None
        if expected is None:
            assert actual is None
        else:
            assert actual == pytest.approx(expected, rel=1e-6)


@pytest.mark.parametrize("window", [0, -1, 2.5, True, "24"])
def test_trailing_mean_rejects_invalid_windows(window):
    with pytest.raises(ValueError, match="positive integer"):
        trailing_mean(pa.array([1.0, 2.0]), window)


def test_trailing_mean_rejects_non_numeric_values():
    with pytest.raises(ValueError, match="must be numeric"):
        trailing_mean(pa.array(["1.0", "2.0"]), 2)


def test_trailing_mean_rejects_non_array_values():
    with pytest.raises(TypeError, match="pyarrow Array or ChunkedArray"):
        trailing_mean([1.0, 2.0], 2)  # type: ignore[arg-type]