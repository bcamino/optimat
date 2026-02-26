"""Tests for v1 YAML config parsing and validation."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from optimat.config import ConfigError, read_yaml
from optimat.config.io import parse_config



def test_read_yaml_parses_dev_input() -> None:
    config = read_yaml(Path("dev_data/input_test.yml"))

    assert config.project.name == "brass_test"
    assert config.structure.file == "test_structure.cif"
    assert len(config.occupancy.site_groups) == 2

    first_group = config.occupancy.site_groups[0]
    assert first_group.indices == [0, 1, 2, 3]
    assert first_group.allowed_species == ["Mg", "Ca"]
    assert first_group.composition.counts == {"Mg": 2, "Ca": 2}

    second_group = config.occupancy.site_groups[1]
    assert sum(second_group.composition.counts.values()) == 4
    assert second_group.allowed_species == ["O"]

    assert config.solver.backend == "cp-sat"
    assert config.solver.cp_sat is not None
    assert config.solver.cp_sat.num_workers == 8



def test_validation_rejects_bad_counts_sum() -> None:
    path = Path("dev_data/input_test.yml")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    data["occupancy"]["site_groups"][0]["composition"]["counts"]["Mg"] = 1

    with pytest.raises(ConfigError, match="counts sum"):
        parse_config(data)
