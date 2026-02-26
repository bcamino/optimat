"""Tests for the exact exhaustive backend."""

from __future__ import annotations

from pathlib import Path

import pytest

from optimat.backends.exact import compile_to_exact, run_exact
from optimat.config import read_yaml
from optimat.energy.terms import EnergyTerms
from optimat.problem import build_problem
from optimat.problem.canonical import CanonicalProblem, CompositionConstraint


def test_exact_backend_tiny_synthetic() -> None:
    terms = EnergyTerms(
        E0=0.0,
        onsite={},
        pair={
            (0, 1, "A", "A"): -2.0,
            (2, 3, "B", "B"): -3.0,
        },
        meta={"model_sites": (0, 1, 2, 3)},
    )
    problem = CanonicalProblem(
        name="toy",
        output_dir=".",
        structure=None,  # type: ignore[arg-type]
        n_sites=4,
        allowed_by_site={0: ("A", "B"), 1: ("A", "B"), 2: ("A", "B"), 3: ("A", "B")},
        variable_sites=(0, 1, 2, 3),
        fixed_assignments={},
        site_groups={"g": (0, 1, 2, 3)},
        composition_constraints=(
            CompositionConstraint(group_name="g", indices=(0, 1, 2, 3), counts={"A": 2, "B": 2}),
        ),
        energy_terms=terms,
        pair_graph={0: (1,), 1: (0,), 2: (3,), 3: (2,)},
        meta={},
    )

    compiled = compile_to_exact(problem)
    result = run_exact(compiled, top_k=3)

    assert result.best_energy == -5.0
    assert result.best_assignment[0] == "A"
    assert result.best_assignment[1] == "A"
    assert result.best_assignment[2] == "B"
    assert result.best_assignment[3] == "B"
    assert result.stats.num_feasible == 6
    assert result.stats.num_evaluated == 6
    assert result.top_k is not None
    assert len(result.top_k) == 3

def test_exact_backend_integration_smoke() -> None:
    pytest.importorskip("pymatgen")

    config, base_dir = read_yaml(Path("dev_data/input_test.yml"), return_base_dir=True)
    problem = build_problem(config, base_dir=base_dir)
    compiled = compile_to_exact(problem)
    result = run_exact(compiled, max_evals=20, top_k=2)

    assert result.stats.num_evaluated > 0
    assert result.stats.num_feasible > 0
    assert isinstance(result.best_assignment, dict)
