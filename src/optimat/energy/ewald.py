"""Ewald term compilation via pymatgen."""

from __future__ import annotations

import math
from itertools import product
from typing import TYPE_CHECKING

from optimat.config import ConfigError

if TYPE_CHECKING:
    from pymatgen.core import Structure


def compute_ewald_baseline_delta_terms(
    structure: "Structure",
    model_sites: list[int],
    allowed_by_site: dict[int, list[str]],
    charges_by_species: dict[str, float],
    ewald_settings: object,
) -> dict[tuple[int, int, str, str], float]:
    """Compile approximate pair terms from Ewald baseline-delta evaluations."""
    try:
        from pymatgen.analysis.ewald import EwaldSummation
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("pymatgen is required for Ewald compilation") from exc

    ordered_sites = sorted(model_sites)
    baseline_assignment = {i: allowed_by_site[i][0] for i in ordered_sites}
    baseline_vector = tuple(baseline_assignment[i] for i in ordered_sites)
    index_to_pos = {site: pos for pos, site in enumerate(ordered_sites)}
    site_count = len(structure)

    def site_symbol(site: object) -> str:
        specie = getattr(site, "specie", None)
        symbol = getattr(specie, "symbol", None)
        if isinstance(symbol, str):
            return symbol
        species_string = getattr(site, "species_string", None)
        if isinstance(species_string, str):
            return species_string
        raise ConfigError("Could not infer species symbol from structure site")

    fixed_species: dict[int, str] = {}
    for idx, site in enumerate(structure):
        if idx in baseline_assignment:
            continue
        symbol = site_symbol(site)
        if symbol not in charges_by_species:
            raise ConfigError(f"Missing charge for fixed structure species {symbol!r}")
        fixed_species[idx] = symbol

    def total_energy_for_assignment(species_vector: tuple[str, ...]) -> float:
        charges: list[float] = [0.0] * site_count
        for idx, symbol in fixed_species.items():
            charges[idx] = charges_by_species[symbol]
        for site in ordered_sites:
            pos = index_to_pos[site]
            symbol = species_vector[pos]
            if symbol not in charges_by_species:
                raise ConfigError(f"Missing charge for species {symbol!r}")
            charges[site] = charges_by_species[symbol]

        structure_copy = structure.copy()
        structure_copy.add_oxidation_state_by_site(charges)

        ewald = EwaldSummation(structure_copy, **_ewald_kwargs(ewald_settings))
        return float(ewald.total_energy)

    cache: dict[tuple[str, ...], float] = {}

    def energy_cached(species_vector: tuple[str, ...]) -> float:
        if species_vector not in cache:
            cache[species_vector] = total_energy_for_assignment(species_vector)
        return cache[species_vector]

    ref_energy = energy_cached(baseline_vector)
    pair_terms: dict[tuple[int, int, str, str], float] = {}

    for left_pos, i in enumerate(ordered_sites):
        for j in ordered_sites[left_pos + 1 :]:
            j_pos = index_to_pos[j]
            for si, sj in product(allowed_by_site[i], allowed_by_site[j]):
                vec = list(baseline_vector)
                vec[left_pos] = si
                vec[j_pos] = sj
                pair_terms[(i, j, si, sj)] = energy_cached(tuple(vec)) - ref_energy

    return pair_terms


def _ewald_kwargs(ewald_settings: object) -> dict[str, float]:
    mode = str(getattr(ewald_settings, "mode", "auto"))
    kwargs: dict[str, float] = {}

    if mode == "manual":
        for key in ("real_space_cut", "recip_space_cut", "eta"):
            value = getattr(ewald_settings, key, None)
            if value is None:
                raise ConfigError(f"energy_model.ewald.{key} is required in manual mode")
            kwargs[key] = float(value)
    else:
        accuracy = getattr(ewald_settings, "accuracy", None)
        if accuracy is not None:
            accuracy_value = float(accuracy)
            if accuracy_value <= 0:
                raise ConfigError("energy_model.ewald.accuracy must be > 0")
            kwargs["acc_factor"] = max(1.0, -math.log10(accuracy_value))

    return kwargs
