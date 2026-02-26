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
        required_sites_meta = self.meta.get("model_sites")
        if isinstance(required_sites_meta, (list, tuple, set)):
            required_sites = sorted(int(i) for i in required_sites_meta)
        else:
            required_set: set[int] = set()
            required_set.update(i for (i, _) in self.onsite)
            required_set.update(i for (i, _, _, _) in self.pair)
            required_set.update(j for (_, j, _, _) in self.pair)
            required_sites = sorted(required_set)

        missing_sites = [i for i in required_sites if i not in assignment]
        if missing_sites:
            raise ValueError(f"Missing assignment for required sites: {missing_sites}")

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
