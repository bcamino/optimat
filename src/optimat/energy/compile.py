"""Energy model compilation entrypoint."""

from __future__ import annotations

from pathlib import Path

from optimat.config import ConfigError, OptimatConfig
from optimat.energy.buckingham import build_buckingham_params, compute_buckingham_pair_terms
from optimat.energy.ewald import compute_ewald_baseline_delta_terms
from optimat.energy.terms import EnergyTerms
from optimat.structures.load import load_structure


def build_allowed_species_by_site(config: OptimatConfig) -> tuple[dict[int, list[str]], list[int]]:
    """Normalize occupancy groups into per-site allowed species."""
    if not config.energy_model.species:
        raise ConfigError("energy_model.species is required for energy compilation")

    known_species = set(config.energy_model.species)
    allowed_by_site: dict[int, list[str]] = {}
    for group in config.occupancy.site_groups:
        invalid = [sp for sp in group.allowed_species if sp not in known_species]
        if invalid:
            raise ConfigError(f"Unknown species in site group {group.name!r}: {invalid}")
        for idx in group.indices:
            if idx in allowed_by_site:
                raise ConfigError(f"Duplicate site index in occupancy groups: {idx}")
            allowed_by_site[idx] = list(group.allowed_species)

    return allowed_by_site, sorted(allowed_by_site)


def compile_energy_model(config: OptimatConfig) -> EnergyTerms:
    """Compile Buckingham + Ewald pair terms from a parsed config."""
    if config.energy_model.type != "buckingham_ewald":
        raise ConfigError(f"Unsupported energy_model.type for compiler: {config.energy_model.type!r}")
    if config.energy_model.buckingham is None or config.energy_model.ewald is None:
        raise ConfigError("energy_model.buckingham and energy_model.ewald are required")

    structure = load_structure(Path(config.structure.file))
    allowed_by_site, model_sites = build_allowed_species_by_site(config)
    if not model_sites:
        raise ConfigError("No model sites found in occupancy.site_groups")
    if max(model_sites) >= len(structure):
        raise ConfigError(
            f"Occupancy site index {max(model_sites)} is out of range for structure with {len(structure)} sites"
        )

    buck_pair = compute_buckingham_pair_terms(
        structure=structure,
        sites=model_sites,
        allowed_by_site=allowed_by_site,
        params=build_buckingham_params(config.energy_model.buckingham.parameters),
        cutoff=float(config.energy_model.buckingham.cutoff),
    )
    charges_by_species = {sp: float(spec.charge) for sp, spec in config.energy_model.species.items()}
    ewald_pair = compute_ewald_baseline_delta_terms(
        structure=structure,
        model_sites=model_sites,
        allowed_by_site=allowed_by_site,
        charges_by_species=charges_by_species,
        ewald_settings=config.energy_model.ewald,
    )

    pair_total = dict(buck_pair)
    for key, value in ewald_pair.items():
        pair_total[key] = pair_total.get(key, 0.0) + value

    meta = {
        "structure_file": config.structure.file,
        "model_site_count": len(model_sites),
        "buckingham_cutoff": float(config.energy_model.buckingham.cutoff),
        "buckingham_neighbor_pairs": len({(i, j) for (i, j, _, _) in buck_pair}),
        "buckingham_pair_entries": len(buck_pair),
        "ewald_pair_entries": len(ewald_pair),
        "ewald_compilation": "baseline_delta_approx_v1",
    }
    return EnergyTerms(E0=0.0, onsite={}, pair=pair_total, meta=meta)
