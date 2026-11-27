"""Numeric helpers shared by ratio-style feature groups."""

from __future__ import annotations

import pyarrow as pa
import pyarrow.compute as pc

__all__ = ["safe_ratio"]

_FLOAT32 = pa.float32()
_ZERO = pa.scalar(0.0, type=_FLOAT32)
_NULL = pa.scalar(None, type=_FLOAT32)
_ARRAY_TYPES = (pa.Array, pa.ChunkedArray)


def safe_ratio(
    numerator: pa.Array | pa.ChunkedArray,
    denominator: pa.Array | pa.ChunkedArray,
) -> pa.Array | pa.ChunkedArray:
    """
    Divides two numeric Arrow arrays, returning a float32 ratio.

    Nulls propagate. A zero denominator (including ``-0.0``) becomes null rather
    than infinity. Both operands are cast to float32 before dividing, so integer
    inputs produce fractional results instead of truncated integer division.
    """
    if not isinstance(numerator, _ARRAY_TYPES) or not isinstance(
        denominator, _ARRAY_TYPES
    ):
        raise TypeError(
            "safe_ratio expects pyarrow Array or ChunkedArray inputs, got "
            f"{type(numerator).__name__} and {type(denominator).__name__}."
        )
    if len(numerator) != len(denominator):
        raise ValueError(
            "Numerator and denominator must have the same length, got "
            f"{len(numerator)} and {len(denominator)}."
        )
    for label, value in (("Numerator", numerator), ("Denominator", denominator)):
        if not _is_numeric(value.type):
            raise ValueError(
                f"{label} must be numeric (integer or floating), got {value.type}."
            )

    numerator32 = pc.cast(numerator, _FLOAT32)
    denominator32 = pc.cast(denominator, _FLOAT32)
    zero = pc.equal(denominator32, _ZERO)
    ratio = pc.if_else(zero, _NULL, pc.divide(numerator32, denominator32))
    return pc.cast(ratio, _FLOAT32)


def _is_numeric(dtype: pa.DataType) -> bool:
    return bool(pa.types.is_integer(dtype) or pa.types.is_floating(dtype))