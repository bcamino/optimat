"""Validation helpers for parsed config objects.

TODO: split validation into schema-level and domain-level passes as rules grow.
"""

from __future__ import annotations

from collections import defaultdict

from .schema import OptimatConfig


class ConfigError(ValueError):
    """Raised when the input configuration is invalid."""



def validate_config(config: OptimatConfig) -> None:
    """Validate v1 configuration constraints."""
    if not config.project.name.strip():
        raise ConfigError("project.name must be a non-empty string")

    if not isinstance(config.structure.file, str) or not config.structure.file.strip():
        raise ConfigError("structure.file must be a non-empty string")
    if not isinstance(config.structure.periodic, bool):
        raise ConfigError("structure.periodic must be a boolean")

    seen_indices: dict[int, list[str]] = defaultdict(list)

    for i, group in enumerate(config.occupancy.site_groups):
        prefix = f"occupancy.site_groups[{i}]"

        if not group.name.strip():
            raise ConfigError(f"{prefix}.name must be a non-empty string")

        if not isinstance(group.indices, list) or not group.indices:
            raise ConfigError(f"{prefix}.indices must be a non-empty list of ints")
        if any((not isinstance(idx, int) or idx < 0) for idx in group.indices):
            raise ConfigError(f"{prefix}.indices must contain only ints >= 0")

        if not isinstance(group.allowed_species, list) or not group.allowed_species:
            raise ConfigError(f"{prefix}.allowed_species must be a non-empty list of strings")
        if any((not isinstance(sp, str) or not sp.strip()) for sp in group.allowed_species):
            raise ConfigError(f"{prefix}.allowed_species must contain only non-empty strings")

        if group.composition.mode != "counts":
            raise ConfigError(f"{prefix}.composition.mode must be 'counts' (v1 only)")

        counts = group.composition.counts
        if not isinstance(counts, dict) or not counts:
            raise ConfigError(f"{prefix}.composition.counts must be a non-empty mapping")
        if any((not isinstance(k, str) or not k.strip()) for k in counts):
            raise ConfigError(f"{prefix}.composition.counts keys must be non-empty strings")
        if any((not isinstance(v, int) or v < 0) for v in counts.values()):
            raise ConfigError(f"{prefix}.composition.counts values must be ints >= 0")

        allowed = set(group.allowed_species)
        count_keys = set(counts)
        invalid_keys = sorted(count_keys - allowed)
        if invalid_keys:
            raise ConfigError(
                f"{prefix}.composition.counts keys must be a subset of allowed_species; invalid keys: {invalid_keys}"
            )

        if sum(counts.values()) != len(group.indices):
            raise ConfigError(
                f"{prefix}.composition.counts sum ({sum(counts.values())}) must equal number of indices ({len(group.indices)})"
            )

        for idx in group.indices:
            seen_indices[idx].append(group.name)

    overlapping = {idx: names for idx, names in seen_indices.items() if len(names) > 1}
    if overlapping:
        parts = [f"{idx}: {names}" for idx, names in sorted(overlapping.items())]
        raise ConfigError("occupancy.site_groups indices must not overlap; duplicates found: " + "; ".join(parts))

    if config.solver.backend == "cp-sat":
        if config.solver.cp_sat is None:
            raise ConfigError("solver.cp_sat is required when solver.backend == 'cp-sat'")
        if config.solver.cp_sat.time_limit <= 0:
            raise ConfigError("solver.cp_sat.time_limit must be > 0")
        if config.solver.cp_sat.num_workers < 1:
            raise ConfigError("solver.cp_sat.num_workers must be >= 1")

    if config.energy_model.type == "buckingham_ewald":
        if not config.energy_model.species:
            raise ConfigError("energy_model.species is required for energy_model.type == 'buckingham_ewald'")
        if config.energy_model.buckingham is None:
            raise ConfigError("energy_model.buckingham is required for energy_model.type == 'buckingham_ewald'")
        if config.energy_model.ewald is None:
            raise ConfigError("energy_model.ewald is required for energy_model.type == 'buckingham_ewald'")

    if config.energy_model.buckingham is not None:
        if config.energy_model.buckingham.cutoff <= 0:
            raise ConfigError("energy_model.buckingham.cutoff must be > 0")
        if not config.energy_model.buckingham.parameters:
            raise ConfigError("energy_model.buckingham.parameters must be non-empty")

    if config.energy_model.ewald is not None:
        if config.energy_model.ewald.engine != "pymatgen":
            raise ConfigError("energy_model.ewald.engine must be 'pymatgen' (v1)")
        if config.energy_model.ewald.mode not in {"auto", "manual"}:
            raise ConfigError("energy_model.ewald.mode must be 'auto' or 'manual'")
        if config.energy_model.ewald.mode == "manual":
            if config.energy_model.ewald.real_space_cut is None:
                raise ConfigError("energy_model.ewald.real_space_cut is required in manual mode")
            if config.energy_model.ewald.recip_space_cut is None:
                raise ConfigError("energy_model.ewald.recip_space_cut is required in manual mode")
            if config.energy_model.ewald.eta is None:
                raise ConfigError("energy_model.ewald.eta is required in manual mode")
