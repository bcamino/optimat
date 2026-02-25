"""YAML configuration reader for v1 optimat input files.

TODO: support schema versions and richer diagnostics with source locations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .schema import (
    BuckinghamConfig,
    BuckinghamPairConfig,
    BuckinghamSmoothingConfig,
    CompositionCounts,
    CpSatOptions,
    EnergyModelConfig,
    EwaldConfig,
    OccupancyConfig,
    OptimatConfig,
    OptimisationConfig,
    ProjectConfig,
    SiteGroupConfig,
    SolverConfig,
    SpeciesConfig,
    StructureConfig,
)
from .validate import ConfigError, validate_config



def read_yaml(path: str | Path) -> OptimatConfig:
    """Read and validate an optimat YAML config file."""
    try:
        import yaml
    except ImportError as exc:
        raise ConfigError("pyyaml is required to read YAML config files") from exc

    file_path = Path(path)
    try:
        with file_path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
    except FileNotFoundError as exc:
        raise ConfigError(f"Config file not found: {file_path}") from exc
    except OSError as exc:
        raise ConfigError(f"Could not read config file {file_path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {file_path}: {exc}") from exc

    return parse_config(raw)



def parse_config(data: Any) -> OptimatConfig:
    """Parse and validate config from a Python mapping."""
    root = _expect_mapping(data, "<root>")

    project_key = "project" if "project" in root else "vproject" if "vproject" in root else "project"
    project_raw = _expect_mapping(_require_key(root, project_key, "<root>"), project_key)
    structure_raw = _expect_mapping(_require_key(root, "structure", "<root>"), "structure")
    occupancy_raw = _expect_mapping(_require_key(root, "occupancy", "<root>"), "occupancy")
    energy_model_raw = _expect_mapping(_require_key(root, "energy_model", "<root>"), "energy_model")
    optimisation_raw = _expect_mapping(_require_key(root, "optimisation", "<root>"), "optimisation")
    solver_raw = _expect_mapping(_require_key(root, "solver", "<root>"), "solver")

    project = ProjectConfig(
        name=_expect_str(_require_key(project_raw, "name", "project"), "project.name"),
        output_dir=_expect_str(_require_key(project_raw, "output_dir", "project"), "project.output_dir"),
    )

    structure = StructureConfig(
        file=_expect_str(_require_key(structure_raw, "file", "structure"), "structure.file"),
        periodic=_expect_bool(_require_key(structure_raw, "periodic", "structure"), "structure.periodic"),
    )

    site_groups_raw = _require_key(occupancy_raw, "site_groups", "occupancy")
    if not isinstance(site_groups_raw, list):
        raise ConfigError("occupancy.site_groups must be a list")
    site_groups: list[SiteGroupConfig] = []
    for idx, item in enumerate(site_groups_raw):
        sg_path = f"occupancy.site_groups[{idx}]"
        sg_raw = _expect_mapping(item, sg_path)
        comp_raw = _expect_mapping(_require_key(sg_raw, "composition", sg_path), f"{sg_path}.composition")
        mode = _expect_str(_require_key(comp_raw, "mode", f"{sg_path}.composition"), f"{sg_path}.composition.mode")
        counts_raw = _expect_mapping(
            _require_key(comp_raw, "counts", f"{sg_path}.composition"), f"{sg_path}.composition.counts"
        )
        counts: dict[str, int] = {}
        for species, value in counts_raw.items():
            if not isinstance(species, str):
                raise ConfigError(f"{sg_path}.composition.counts keys must be strings")
            if not isinstance(value, int) or isinstance(value, bool):
                raise ConfigError(f"{sg_path}.composition.counts[{species!r}] must be an int")
            counts[species] = value

        indices_raw = _require_key(sg_raw, "indices", sg_path)
        if not isinstance(indices_raw, list):
            raise ConfigError(f"{sg_path}.indices must be a list")
        indices: list[int] = []
        for j, value in enumerate(indices_raw):
            if not isinstance(value, int) or isinstance(value, bool):
                raise ConfigError(f"{sg_path}.indices[{j}] must be an int")
            indices.append(value)

        allowed_raw = _require_key(sg_raw, "allowed_species", sg_path)
        if not isinstance(allowed_raw, list):
            raise ConfigError(f"{sg_path}.allowed_species must be a list")
        allowed_species: list[str] = []
        for j, value in enumerate(allowed_raw):
            if not isinstance(value, str):
                raise ConfigError(f"{sg_path}.allowed_species[{j}] must be a string")
            allowed_species.append(value)

        site_groups.append(
            SiteGroupConfig(
                name=_expect_str(_require_key(sg_raw, "name", sg_path), f"{sg_path}.name"),
                indices=indices,
                allowed_species=allowed_species,
                composition=CompositionCounts(mode=mode, counts=counts),
            )
        )

    occupancy = OccupancyConfig(site_groups=site_groups)

    energy_model = EnergyModelConfig(
        type=_expect_str(_require_key(energy_model_raw, "type", "energy_model"), "energy_model.type"),
        parameters_file=(
            _expect_str(energy_model_raw["parameters_file"], "energy_model.parameters_file")
            if "parameters_file" in energy_model_raw and energy_model_raw["parameters_file"] is not None
            else None
        ),
        species=_parse_species(energy_model_raw.get("species")),
        buckingham=_parse_buckingham(energy_model_raw.get("buckingham")),
        ewald=_parse_ewald(energy_model_raw.get("ewald")),
    )

    optimisation = OptimisationConfig(
        objective=_expect_str(_require_key(optimisation_raw, "objective", "optimisation"), "optimisation.objective")
    )

    cp_sat: CpSatOptions | None = None
    backend = _expect_str(_require_key(solver_raw, "backend", "solver"), "solver.backend")
    if "cp_sat" in solver_raw and solver_raw["cp_sat"] is not None:
        cp_sat_raw = _expect_mapping(solver_raw["cp_sat"], "solver.cp_sat")
        time_limit = _require_key(cp_sat_raw, "time_limit", "solver.cp_sat")
        if not isinstance(time_limit, (int, float)) or isinstance(time_limit, bool):
            raise ConfigError("solver.cp_sat.time_limit must be an int or float")
        num_workers = _require_key(cp_sat_raw, "num_workers", "solver.cp_sat")
        if not isinstance(num_workers, int) or isinstance(num_workers, bool):
            raise ConfigError("solver.cp_sat.num_workers must be an int")
        cp_sat = CpSatOptions(time_limit=time_limit, num_workers=num_workers)

    solver = SolverConfig(backend=backend, cp_sat=cp_sat)

    config = OptimatConfig(
        project=project,
        structure=structure,
        occupancy=occupancy,
        energy_model=energy_model,
        optimisation=optimisation,
        solver=solver,
    )
    validate_config(config)
    return config



def _require_key(mapping: dict[str, Any], key: str, path: str) -> Any:
    if key not in mapping:
        raise ConfigError(f"Missing required key: {path}.{key}" if path != "<root>" else f"Missing required key: {key}")
    return mapping[key]



def _expect_mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"{path} must be a mapping")
    for key in value:
        if not isinstance(key, str):
            raise ConfigError(f"{path} must have string keys")
    return value



def _expect_str(value: Any, path: str) -> str:
    if not isinstance(value, str):
        raise ConfigError(f"{path} must be a string")
    return value



def _expect_bool(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{path} must be a boolean")
    return value


def _expect_number(value: Any, path: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ConfigError(f"{path} must be a number")
    return float(value)


def _parse_species(value: Any) -> dict[str, SpeciesConfig]:
    if value is None:
        return {}
    raw = _expect_mapping(value, "energy_model.species")
    species: dict[str, SpeciesConfig] = {}
    for name, entry in raw.items():
        entry_raw = _expect_mapping(entry, f"energy_model.species[{name!r}]")
        species[name] = SpeciesConfig(
            charge=_expect_number(
                _require_key(entry_raw, "charge", f"energy_model.species[{name!r}]"),
                f"energy_model.species[{name!r}].charge",
            )
        )
    return species


def _parse_buckingham(value: Any) -> BuckinghamConfig | None:
    if value is None:
        return None
    raw = _expect_mapping(value, "energy_model.buckingham")
    smoothing: BuckinghamSmoothingConfig | None = None
    if "smoothing" in raw and raw["smoothing"] is not None:
        smoothing_raw = _expect_mapping(raw["smoothing"], "energy_model.buckingham.smoothing")
        smoothing = BuckinghamSmoothingConfig(
            enabled=_expect_bool(
                _require_key(smoothing_raw, "enabled", "energy_model.buckingham.smoothing"),
                "energy_model.buckingham.smoothing.enabled",
            )
        )

    parameters_raw = _expect_mapping(_require_key(raw, "parameters", "energy_model.buckingham"), "energy_model.buckingham.parameters")
    parameters: dict[str, BuckinghamPairConfig] = {}
    for pair_key, pair_value in parameters_raw.items():
        pair_raw = _expect_mapping(pair_value, f"energy_model.buckingham.parameters[{pair_key!r}]")
        parameters[pair_key] = BuckinghamPairConfig(
            A=_expect_number(_require_key(pair_raw, "A", f"energy_model.buckingham.parameters[{pair_key!r}]"), f"energy_model.buckingham.parameters[{pair_key!r}].A"),
            rho=_expect_number(_require_key(pair_raw, "rho", f"energy_model.buckingham.parameters[{pair_key!r}]"), f"energy_model.buckingham.parameters[{pair_key!r}].rho"),
            C=_expect_number(_require_key(pair_raw, "C", f"energy_model.buckingham.parameters[{pair_key!r}]"), f"energy_model.buckingham.parameters[{pair_key!r}].C"),
        )

    return BuckinghamConfig(
        cutoff=_expect_number(_require_key(raw, "cutoff", "energy_model.buckingham"), "energy_model.buckingham.cutoff"),
        smoothing=smoothing,
        parameters=parameters,
    )


def _parse_ewald(value: Any) -> EwaldConfig | None:
    if value is None:
        return None
    raw = _expect_mapping(value, "energy_model.ewald")
    return EwaldConfig(
        engine=_expect_str(_require_key(raw, "engine", "energy_model.ewald"), "energy_model.ewald.engine"),
        mode=_expect_str(_require_key(raw, "mode", "energy_model.ewald"), "energy_model.ewald.mode"),
        accuracy=(_expect_number(raw["accuracy"], "energy_model.ewald.accuracy") if "accuracy" in raw and raw["accuracy"] is not None else None),
        real_space_cut=(
            _expect_number(raw["real_space_cut"], "energy_model.ewald.real_space_cut")
            if "real_space_cut" in raw and raw["real_space_cut"] is not None
            else None
        ),
        recip_space_cut=(
            _expect_number(raw["recip_space_cut"], "energy_model.ewald.recip_space_cut")
            if "recip_space_cut" in raw and raw["recip_space_cut"] is not None
            else None
        ),
        eta=(_expect_number(raw["eta"], "energy_model.ewald.eta") if "eta" in raw and raw["eta"] is not None else None),
    )
