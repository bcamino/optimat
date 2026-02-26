"""Build a canonical, solver-agnostic problem from config."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from optimat.config import ConfigError, OptimatConfig
from optimat.energy import compile_energy_model
from optimat.problem.canonical import CanonicalProblem, CompositionConstraint
from optimat.structures import load_structure
from optimat.util.paths import ensure_dir, resolve_path


def build_problem(config: OptimatConfig, *, base_dir: Path | None = None) -> CanonicalProblem:
    """Build the canonical problem object from a parsed config."""
    base = Path.cwd() if base_dir is None else Path(base_dir)

    structure_path = resolve_path(config.structure.file, base)
    structure = load_structure(structure_path)
    n_sites = len(structure)

    output_dir = ensure_dir(resolve_path(config.project.output_dir, base))

    allowed_by_site_lists: dict[int, list[str]] = {}
    site_groups: dict[str, tuple[int, ...]] = {}
    index_to_group: dict[int, str] = {}
    composition_constraints: list[CompositionConstraint] = []

    for group in config.occupancy.site_groups:
        indices = tuple(group.indices)
        site_groups[group.name] = indices

        for i in indices:
            if i < 0 or i >= n_sites:
                raise ConfigError(
                    f"occupancy.site_groups[{group.name!r}] index {i} is out of range for structure with {n_sites} sites"
                )
            if i in allowed_by_site_lists:
                prev_group = index_to_group[i]
                raise ConfigError(f"Index {i} appears in multiple groups: {prev_group}, {group.name}")
            allowed_by_site_lists[i] = list(group.allowed_species)
            index_to_group[i] = group.name

        if group.composition.mode != "counts":
            raise ConfigError(f"Unsupported composition.mode {group.composition.mode!r} for group {group.name!r}")
        counts = dict(group.composition.counts)
        if sum(counts.values()) != len(indices):
            raise ConfigError(
                f"Composition counts for group {group.name!r} sum to {sum(counts.values())}, "
                f"expected {len(indices)}"
            )
        invalid = sorted(set(counts) - set(group.allowed_species))
        if invalid:
            raise ConfigError(
                f"Composition counts for group {group.name!r} include species not allowed in group: {invalid}"
            )
        composition_constraints.append(
            CompositionConstraint(group_name=group.name, indices=indices, counts=counts)
        )

    allowed_by_site = {i: tuple(species) for i, species in sorted(allowed_by_site_lists.items())}
    fixed_assignments = {i: allowed[0] for i, allowed in allowed_by_site.items() if len(allowed) == 1}
    variable_sites = tuple(sorted(i for i, allowed in allowed_by_site.items() if len(allowed) > 1))

    energy_terms = compile_energy_model(
        config,
        base_dir=base,
        structure=structure,
        allowed_by_site=allowed_by_site_lists,
    )

    adjacency: dict[int, set[int]] = {i: set() for i in allowed_by_site}
    for i, j, _, _ in energy_terms.pair:
        adjacency.setdefault(i, set()).add(j)
        adjacency.setdefault(j, set()).add(i)
    pair_graph = {i: tuple(sorted(neigh)) for i, neigh in sorted(adjacency.items())}

    cutoff = None
    if config.energy_model.buckingham is not None:
        cutoff = float(config.energy_model.buckingham.cutoff)

    meta: dict[str, Any] = {
        "structure_path": str(structure_path),
        "energy_model_type": config.energy_model.type,
        "buckingham_cutoff": cutoff,
        "energy_meta": dict(energy_terms.meta),
    }

    return CanonicalProblem(
        name=config.project.name,
        output_dir=str(output_dir),
        structure=structure,
        n_sites=n_sites,
        allowed_by_site=allowed_by_site,
        variable_sites=variable_sites,
        fixed_assignments=fixed_assignments,
        site_groups=site_groups,
        composition_constraints=tuple(composition_constraints),
        energy_terms=energy_terms,
        pair_graph=pair_graph,
        meta=meta,
    )
