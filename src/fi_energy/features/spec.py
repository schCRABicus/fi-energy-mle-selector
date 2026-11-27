"""Immutable metadata describing one derived model feature."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["FeatureSpec"]


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    """
    Declares a derived feature's output name, inputs, group, and purpose.

    A spec is metadata only: the computation lives in the feature module that
    owns the group. ``inputs`` may name raw Phase 1 columns or columns produced
    by an earlier feature group.
    """

    name: str
    inputs: tuple[str, ...]
    group: str
    description: str

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Feature name must not be empty.")
        if not self.name.isidentifier():
            raise ValueError(
                f"Feature name {self.name!r} must be a valid Python identifier."
            )
        if isinstance(self.inputs, str):
            raise ValueError(
                f"Feature {self.name!r} inputs must be a sequence of column "
                "names, not a string."
            )

        inputs = tuple(self.inputs)
        if not inputs:
            raise ValueError(
                f"Feature {self.name!r} must declare at least one input."
            )
        if len(set(inputs)) != len(inputs):
            raise ValueError(f"Feature {self.name!r} has duplicate inputs.")
        if self.name in inputs:
            raise ValueError(
                f"Feature {self.name!r} cannot use itself as an input."
            )
        object.__setattr__(self, "inputs", inputs)

        if not self.group.strip():
            raise ValueError(
                f"Feature {self.name!r} must declare a non-empty group."
            )
        if not self.description.strip():
            raise ValueError(
                f"Feature {self.name!r} must declare a non-empty description."
            )