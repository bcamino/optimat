"""Buckingham pair term compilation."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from optimat.config import ConfigError

if TYPE_CHECKING:
    from pymatgen.core import Structure


BuckinghamParams = tuple[float, float, float]


def parse_pair_key(value: str) -> tuple[str, str]:
    """Parse a pair key like 'Mg-O' into a normalized species tuple."""
    parts = [part.strip() for part in value.split("-")]
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ConfigError(f"Invalid Buckingham pair key: {value!r}")
    a, b = parts
    return (a, b) if a <= b else (b, a)


def build_buckingham_params(raw_params: Mapping[str, object]) -> dict[tuple[str, str], BuckinghamParams]:
    """Normalize Buckingham parameters keyed by unordered species pairs."""
    out: dict[tuple[str, str], BuckinghamParams] = {}
    for pair_key, params in raw_params.items():
        if hasattr(params, "A") and hasattr(params, "rho") and hasattr(params, "C"):
            A = float(getattr(params, "A"))
            rho = float(getattr(params, "rho"))
            C = float(getattr(params, "C"))
        elif isinstance(params, Mapping):
            try:
                A = float(params["A"])
                rho = float(params["rho"])
                C = float(params["C"])
            except KeyError as exc:
                raise ConfigError(f"Invalid Buckingham parameters for pair {pair_key!r}") from exc
        else:
            raise ConfigError(f"Invalid Buckingham parameters for pair {pair_key!r}")
        out[parse_pair_key(pair_key)] = (A, rho, C)
    return out


def buckingham_energy(r: float, A: float, rho: float, C: float) -> float:
    """Compute Buckingham pair energy A*exp(-r/rho) - C/r^6."""
    if r == 0:
        raise ConfigError("Buckingham distance r must be non-zero")
    if rho == 0:
        raise ConfigError("Buckingham parameter rho must be non-zero")
    return A * math.exp(-r / rho) - C / (r**6)


def compute_buckingham_pair_terms(
    structure: "Structure",
    sites: Sequence[int],
    allowed_by_site: Mapping[int, Sequence[str]],
    params: Mapping[tuple[str, str], BuckinghamParams],
    cutoff: float,
) -> dict[tuple[int, int, str, str], float]:
    """Compile Buckingham pair terms for model sites within the cutoff."""
    site_set = set(sites)
    terms: dict[tuple[int, int, str, str], float] = {}
    required_species_pairs: set[tuple[str, str]] = set()
    missing_species_pairs: set[tuple[str, str]] = set()

    center_indices, point_indices, _, distances = structure.get_neighbor_list(cutoff)
    for center_idx, point_idx, dist in zip(center_indices, point_indices, distances):
        i = int(center_idx)
        j = int(point_idx)
        if i >= j:
            continue
        if i not in site_set or j not in site_set:
            continue

        r = float(dist)
        for si in allowed_by_site[i]:
            for sj in allowed_by_site[j]:
                pair_species = (si, sj) if si <= sj else (sj, si)
                required_species_pairs.add(pair_species)
                if pair_species not in params:
                    missing_species_pairs.add(pair_species)
                    continue
                A, rho, C = params[pair_species]
                key = (i, j, si, sj)
                terms[key] = terms.get(key, 0.0) + buckingham_energy(r, A, rho, C)

    present_species_pairs = sorted(required_species_pairs - missing_species_pairs)
    missing_species_pairs_sorted = sorted(missing_species_pairs)
    print(f"Buckingham potentials present (used): {present_species_pairs}")
    print(f"Buckingham potentials missing (treated as 0): {missing_species_pairs_sorted}")

    return terms
