"""The FeatureCatalog domain object: an ordered, validated set of groups."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pyarrow as pa

from fi_energy.features.group import FeatureGroup
from fi_energy.features.groups import (
    CalendarFeatureGroup,
    InteractionFeatureGroup,
    ThermalDemandFeatureGroup,
    WindRatiosFeatureGroup,
)
from fi_energy.features.spec import FeatureSpec

__all__ = ["DEFAULT_CATALOG", "FeatureCatalog"]


@dataclass(frozen=True, slots=True)
class FeatureCatalog:
    """
    An ordered collection of feature groups that can be computed together.

    Groups run in declaration order. A group may consume raw Phase 1 columns
    or features produced by any group listed before it; construction rejects
    duplicate group or feature names and out-of-order dependencies, so an
    invalid combination cannot reach the pipeline.
    """

    groups: tuple[FeatureGroup, ...]

    def __post_init__(self) -> None:
        groups = tuple(self.groups)
        object.__setattr__(self, "groups", groups)
        _validate_groups(groups)

    @property
    def specs(self) -> tuple[FeatureSpec, ...]:
        """Returns every feature spec in group and declaration order."""
        return tuple(spec for group in self.groups for spec in group.specs)

    @property
    def feature_names(self) -> tuple[str, ...]:
        """Returns every feature name in application order."""
        return tuple(spec.name for spec in self.specs)

    def group(self, name: str) -> FeatureGroup:
        """Returns the group with ``name``, or raises ``KeyError``."""
        for group in self.groups:
            if group.name == name:
                return group
        raise KeyError(f"No feature group named {name!r}.")

    def subset(self, *names: str) -> FeatureCatalog:
        """
        Returns a catalog of the named groups, in this catalog's order.

        Useful for feature-ablation tiers. The subset is validated like any
        other catalog, so omitting a group whose features another selected
        group consumes raises ``ValueError``.
        """
        if not names:
            raise ValueError("Select at least one feature group.")
        requested = set(names)
        known = {group.name for group in self.groups}
        missing = sorted(requested - known)
        if missing:
            raise KeyError(
                f"Unknown feature group(s): {', '.join(missing)}."
            )
        selected = tuple(group for group in self.groups if group.name in requested)
        _validate_dependencies(selected, self.feature_names)
        return FeatureCatalog(selected)

    def compute(self, table: pa.Table) -> pa.Table:
        """Applies every group to ``table`` in order."""
        for group in self.groups:
            table = group.compute(table)
        return table

    def diagnostics(self, table: pa.Table) -> dict[str, dict[str, Any]]:
        """Returns each group's diagnostics, keyed by group name."""
        return {
            group.name: group.diagnostics(table) for group in self.groups
        }


def _validate_groups(groups: tuple[FeatureGroup, ...]) -> None:
    if not groups:
        raise ValueError("FeatureCatalog must contain at least one group.")

    group_names: set[str] = set()
    for group in groups:
        if not isinstance(group, FeatureGroup):
            raise TypeError(
                "FeatureCatalog groups must be FeatureGroup values, got "
                f"{type(group).__name__}."
            )
        if group.name in group_names:
            raise ValueError(
                f"Duplicate feature group name {group.name!r}."
            )
        group_names.add(group.name)

    feature_names: set[str] = set()
    for group in groups:
        for name in group.feature_names:
            if name in feature_names:
                raise ValueError(
                    f"Feature {name!r} is produced by more than one group."
                )
            feature_names.add(name)

    _validate_dependencies(groups, feature_names)


def _validate_dependencies(
    groups: tuple[FeatureGroup, ...],
    known_features: set[str],
) -> None:
    """Rejects a group that consumes a known feature before it is produced."""
    produced: set[str] = set()
    for group in groups:
        for name in group.inputs:
            if name in known_features and name not in produced:
                raise ValueError(
                    f"Feature group {group.name!r} requires feature {name!r} "
                    "before it is produced."
                )
        produced.update(group.feature_names)


DEFAULT_CATALOG = FeatureCatalog(
    (
        WindRatiosFeatureGroup(),
        ThermalDemandFeatureGroup(),
        InteractionFeatureGroup(),
        CalendarFeatureGroup(),
    )
)