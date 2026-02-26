"""Ewald term compilation via pymatgen."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from optimat.config import ConfigError

if TYPE_CHECKING:
    from pymatgen.core import Structure


def compute_ewald_pair_terms_from_total_energy_matrix(
    structure: "Structure",
    model_sites: list[int],
    allowed_by_site: dict[int, list[str]],
    charges_by_species: dict[str, float],
    ewald_settings: object,
) -> dict[tuple[int, int, str, str], float]:
    """Compile Ewald pair terms from pymatgen's total_energy_matrix."""
    try:
        from pymatgen.analysis.ewald import EwaldSummation
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("pymatgen is required for Ewald compilation") from exc

    ordered_sites = sorted(model_sites)
    structure_tmp = structure.copy()
    structure_tmp.add_oxidation_state_by_site([1.0] * len(structure_tmp))

    ewald_kwargs = _ewald_kwargs(ewald_settings)
    ewald = EwaldSummation(structure_tmp, w=1, **ewald_kwargs)
    ewald_matrix = np.asarray(ewald.total_energy_matrix, dtype=float)
    ewald_matrix = np.triu(ewald_matrix, 1)

    pair_terms: dict[tuple[int, int, str, str], float] = {}

    for left_pos, i in enumerate(ordered_sites):
        for j in ordered_sites[left_pos + 1 :]:
            kij = float(ewald_matrix[i, j])
            for si in allowed_by_site[i]:
                qi = charges_by_species.get(si)
                if qi is None:
                    raise ConfigError(f"Missing charge for species {si!r}")
                for sj in allowed_by_site[j]:
                    qj = charges_by_species.get(sj)
                    if qj is None:
                        raise ConfigError(f"Missing charge for species {sj!r}")
                    pair_terms[(i, j, si, sj)] = kij * float(qi) * float(qj)

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
