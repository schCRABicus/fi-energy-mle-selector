"""The FeatureGroup domain object: a named, validated bundle of features."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

import pyarrow as pa

from fi_energy.features.spec import FeatureSpec

__all__ = ["FeatureGroup"]


class FeatureGroup(ABC):
    """
    A named bundle of derived features that are computed and validated together.

    Concrete groups declare ``name`` and ``specs`` as class attributes and
    implement ``_compute``. The public ``compute`` method validates the input,
    delegates the transformation, and validates the result, so a subclass
    cannot forget either half of the contract.
    """

    name: ClassVar[str]
    specs: ClassVar[tuple[FeatureSpec, ...]]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        _validate_declaration(cls)

    @property
    def feature_names(self) -> tuple[str, ...]:
        """Returns this group's feature names in application order."""
        return tuple(spec.name for spec in self.specs)

    @property
    def inputs(self) -> tuple[str, ...]:
        """Returns the deduplicated input columns declared by the specs."""
        return tuple(
            dict.fromkeys(name for spec in self.specs for name in spec.inputs)
        )

    def compute(self, table: pa.Table) -> pa.Table:
        """
        Appends this group's feature columns to ``table``.

        The result has exactly the source columns followed by
        ``feature_names``, keeps every row, and stores each feature as a
        nullable ``float32``.

        Raises:
            TypeError: If ``table`` is not a pyarrow Table.
            ValueError: If a required input is missing or invalid, or if an
                output column already exists.
            RuntimeError: If ``_compute`` violates the output contract.
        """
        _require_table(table)
        self.validate_inputs(table)
        result = self._compute(table)
        _validate_output(self, table, result)
        return result

    def validate_inputs(self, table: pa.Table) -> None:
        """
        Checks that inputs exist and output names are free.

        Subclasses extend this to add group-specific checks, such as numeric
        dtype requirements.
        """
        self._require_columns(table, self.inputs)
        present = [
            name for name in self.feature_names if name in table.column_names
        ]
        if present:
            raise ValueError(
                f"Table already has feature column(s): {', '.join(present)}."
            )

    @abstractmethod
    def _compute(self, table: pa.Table) -> pa.Table:
        """Returns ``table`` with this group's feature columns appended."""

    def diagnostics(self, table: pa.Table) -> dict[str, Any]:
        """Summarises corrections or data-quality facts; empty by default."""
        return {}

    @staticmethod
    def _require_columns(table: pa.Table, names: tuple[str, ...]) -> None:
        missing = [name for name in names if name not in table.column_names]
        if missing:
            raise ValueError(
                f"Table is missing required column(s): {', '.join(missing)}."
            )

    @staticmethod
    def _require_numeric(table: pa.Table, name: str) -> None:
        dtype = table.schema.field(name).type
        if not (pa.types.is_integer(dtype) or pa.types.is_floating(dtype)):
            raise ValueError(
                f"Column {name!r} must be numeric (integer or floating), got "
                f"{dtype}."
            )


def _validate_declaration(cls: type[FeatureGroup]) -> None:
    name = cls.__dict__.get("name")
    if not isinstance(name, str) or not name.strip():
        raise TypeError(
            f"{cls.__name__} must declare a non-empty string 'name'."
        )
    if not name.isidentifier():
        raise TypeError(
            f"{cls.__name__} name {name!r} must be a valid Python identifier."
        )

    specs = cls.__dict__.get("specs")
    if not isinstance(specs, tuple) or not specs:
        raise TypeError(
            f"{cls.__name__} must declare a non-empty tuple 'specs' of "
            "FeatureSpec values."
        )

    seen: set[str] = set()
    for spec in specs:
        if not isinstance(spec, FeatureSpec):
            raise TypeError(
                f"{cls.__name__} specs must be FeatureSpec values, got "
                f"{type(spec).__name__}."
            )
        if spec.name in seen:
            raise TypeError(
                f"{cls.__name__} declares duplicate feature {spec.name!r}."
            )
        seen.add(spec.name)


def _validate_output(
    group: FeatureGroup,
    source: pa.Table,
    result: pa.Table,
) -> None:
    expected = [*source.column_names, *group.feature_names]
    if result.column_names != expected:
        raise RuntimeError(
            f"Feature group {group.name!r} must append exactly "
            f"{list(group.feature_names)}, got columns "
            f"{result.column_names}."
        )
    if result.num_rows != source.num_rows:
        raise RuntimeError(
            f"Feature group {group.name!r} changed the row count from "
            f"{source.num_rows} to {result.num_rows}."
        )
    for name in group.feature_names:
        field = result.schema.field(name)
        if not field.type.equals(pa.float32()) or not field.nullable:
            raise RuntimeError(
                f"Feature {name!r} from {group.name!r} must be a nullable "
                f"float32, got {field.type} (nullable={field.nullable})."
            )


def _require_table(value: object) -> None:
    if not isinstance(value, pa.Table):
        raise TypeError(
            f"Expected a pyarrow Table, got {type(value).__name__}."
        )