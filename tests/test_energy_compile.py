"""Smoke tests for Buckingham+Ewald energy compilation."""

from __future__ import annotations

from pathlib import Path

import pytest

from optimat.config import read_yaml
from optimat.energy import compile_energy_model


pytest.importorskip("pymatgen")


def test_compile_energy_model_smoke() -> None:
    config, base_dir = read_yaml(Path("dev_data/input_test.yml"), return_base_dir=True)
    terms = compile_energy_model(config, base_dir=base_dir)

    assert len(terms.pair) > 0

    allowed_by_site: dict[int, set[str]] = {}
    for group in config.occupancy.site_groups:
        for idx in group.indices:
            allowed_by_site[idx] = set(group.allowed_species)

    for i, j, si, sj in terms.pair:
        assert i < j
        assert si in allowed_by_site[i]
        assert sj in allowed_by_site[j]
