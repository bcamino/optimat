"""Tests for canonical problem construction."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from optimat.config import ConfigError, read_yaml
from optimat.config.schema import CompositionCounts, OccupancyConfig, SiteGroupConfig
from optimat.problem import build_problem


pytest.importorskip("pymatgen")


def test_build_problem_smoke(tmp_path: Path) -> None:
    config, base_dir = read_yaml(Path("dev_data/input_test.yml"), return_base_dir=True)
    config = replace(config, project=replace(config.project, output_dir=str(tmp_path / "results")))

    problem = build_problem(config, base_dir=base_dir)

    assert problem.name == "brass_test"
    assert problem.n_sites == len(problem.structure)
    assert "substitutional_bulk" in problem.site_groups
    assert "fixed_sites" in problem.site_groups
    assert problem.variable_sites == (0, 1, 2, 3)
    assert problem.fixed_assignments == {4: "O", 5: "O", 6: "O", 7: "O"}

    counts_by_group = {c.group_name: c.counts for c in problem.composition_constraints}
    assert sum(counts_by_group["substitutional_bulk"].values()) == len(problem.site_groups["substitutional_bulk"])
    assert sum(counts_by_group["fixed_sites"].values()) == len(problem.site_groups["fixed_sites"])


def test_build_problem_rejects_overlap_indices(tmp_path: Path) -> None:
    config, base_dir = read_yaml(Path("dev_data/input_test.yml"), return_base_dir=True)
    config = replace(config, project=replace(config.project, output_dir=str(tmp_path / "results")))

    overlapping_group = SiteGroupConfig(
        name="overlap_group",
        indices=[0],
        allowed_species=["O"],
        composition=CompositionCounts(mode="counts", counts={"O": 1}),
    )
    bad_occupancy = OccupancyConfig(
        site_groups=[config.occupancy.site_groups[0], overlapping_group, config.occupancy.site_groups[1]]
    )
    bad_config = replace(config, occupancy=bad_occupancy)

    with pytest.raises(ConfigError, match=r"Index 0 appears in multiple groups: substitutional_bulk, overlap_group"):
        build_problem(bad_config, base_dir=base_dir)
