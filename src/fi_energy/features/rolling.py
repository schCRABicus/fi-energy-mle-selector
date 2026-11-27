"""Sliding-window helpers for time-ordered feature columns."""

from __future__ import annotations

import pyarrow as pa
import pyarrow.compute as pc

__all__ = ["trailing_mean"]

_FLOAT32 = pa.float32()
_FLOAT64 = pa.float64()
_ZERO64 = pa.scalar(0.0, type=_FLOAT64)
_ZERO_INT = pa.scalar(0, type=pa.int64())
_NULL64 = pa.scalar(None, type=_FLOAT64)
_ARRAY_TYPES = (pa.Array, pa.ChunkedArray)


def trailing_mean(
    values: pa.Array | pa.ChunkedArray,
    window: int,
) -> pa.Array:
    """
    Mean of the ``window`` immediately preceding values, per row.

    The window is right-aligned and ends at the current row, so a feature
    never uses observations from the future. A row is null until ``window``
    rows exist at the start of the series. Null values inside a window are
    skipped and the mean is taken over the remaining observations; a window
    with no observations yields null. The result is ``float32`` and has the
    same length as the input.
    """
    if not isinstance(values, _ARRAY_TYPES):
        raise TypeError(
            "trailing_mean expects a pyarrow Array or ChunkedArray, got "
            f"{type(values).__name__}."
        )
    if isinstance(window, bool) or not isinstance(window, int) or window < 1:
        raise ValueError(
            f"window must be a positive integer, got {window!r}."
        )
    dtype = values.type
    if not (pa.types.is_integer(dtype) or pa.types.is_floating(dtype)):
        raise ValueError(
            f"values must be numeric (integer or floating), got {dtype}."
        )

    if isinstance(values, pa.ChunkedArray):
        values = values.combine_chunks()
    data = pc.cast(values, _FLOAT64)
    rows = len(data)
    if rows < window:
        return pa.nulls(rows, type=_FLOAT32)

    filled = pc.fill_null(data, _ZERO64)
    sums = pc.cumulative_sum(filled)
    counts = pc.cumulative_sum(pc.cast(pc.is_valid(data), pa.int64()))

    # Row i's window covers [i - window + 1, i]; subtracting the cumulative
    # totals through row i - window leaves exactly those rows.
    zero_sums = pc.fill_null(pa.nulls(window, type=_FLOAT64), _ZERO64)
    zero_counts = pc.fill_null(pa.nulls(window, type=pa.int64()), _ZERO_INT)
    window_sums = pc.subtract(
        sums,
        pa.concat_arrays([zero_sums, sums[:-window]]),
    )
    window_counts = pc.subtract(
        counts,
        pa.concat_arrays([zero_counts, counts[:-window]]),
    )

    has_observations = pc.greater(window_counts, _ZERO_INT)
    full_window = pa.array(
        [False] * (window - 1) + [True] * (rows - window + 1),
        type=pa.bool_(),
    )
    means = pc.divide(window_sums, pc.cast(window_counts, _FLOAT64))
    result = pc.if_else(
        pc.and_(full_window, has_observations),
        means,
        _NULL64,
    )
    return pc.cast(result, _FLOAT32)