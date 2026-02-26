"""Canonical solver-agnostic problem dataclasses."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from optimat.config import ConfigError
from optimat.energy.terms import EnergyTerms

if TYPE_CHECKING:
    from pymatgen.core import Structure


@dataclass(frozen=True)
class CompositionConstraint:
    group_name: str
    indices: tuple[int, ...]
    counts: dict[str, int]


@dataclass(frozen=True)
class CanonicalProblem:
    name: str
    output_dir: str
    structure: "Structure"
    n_sites: int
    allowed_by_site: dict[int, tuple[str, ...]]
    variable_sites: tuple[int, ...]
    fixed_assignments: dict[int, str]
    site_groups: dict[str, tuple[int, ...]]
    composition_constraints: tuple[CompositionConstraint, ...]
    energy_terms: EnergyTerms
    pair_graph: dict[int, tuple[int, ...]]
    meta: dict[str, Any] = field(default_factory=dict)

    def check_assignment(self, assignment: dict[int, str]) -> None:
        """Validate a concrete assignment against allowed species and composition counts."""
        missing = [i for i in self.allowed_by_site if i not in assignment]
        if missing:
            raise ConfigError(f"Missing assignments for modeled sites: {sorted(missing)}")

        for i, allowed in self.allowed_by_site.items():
            species = assignment[i]
            if species not in allowed:
                raise ConfigError(f"Species {species!r} not allowed on site {i}; allowed={list(allowed)}")

        for i, species in self.fixed_assignments.items():
            if assignment.get(i) != species:
                raise ConfigError(f"Fixed site {i} must be assigned {species!r}")

        for constraint in self.composition_constraints:
            observed = Counter(assignment[i] for i in constraint.indices)
            species_keys = set(observed) | set(constraint.counts)
            mismatch = {sp: (constraint.counts.get(sp, 0), observed.get(sp, 0)) for sp in species_keys if constraint.counts.get(sp, 0) != observed.get(sp, 0)}
            if mismatch:
                raise ConfigError(
                    f"Composition mismatch for group {constraint.group_name!r}: "
                    f"expected {constraint.counts}, observed {dict(observed)}"
                )
