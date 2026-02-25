"""Compiled energy term containers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class EnergyTerms:
    E0: float
    onsite: dict[tuple[int, str], float] = field(default_factory=dict)
    pair: dict[tuple[int, int, str, str], float] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    def energy_of_assignment(self, assignment: dict[int, str]) -> float:
        """Evaluate the compiled energy for a concrete site->species assignment."""
        total = float(self.E0)

        for site, species in assignment.items():
            total += self.onsite.get((site, species), 0.0)

        sites = sorted(assignment)
        for pos, i in enumerate(sites):
            si = assignment[i]
            for j in sites[pos + 1 :]:
                sj = assignment[j]
                total += self.pair.get((i, j, si, sj), 0.0)

        return total
