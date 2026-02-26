"""Compilation for the exact exhaustive backend."""

from __future__ import annotations

from dataclasses import dataclass

from optimat.config import ConfigError
from optimat.problem import CanonicalProblem


@dataclass(frozen=True)
class ExactGroup:
    name: str
    indices: tuple[int, ...]
    counts: dict[str, int]


@dataclass(frozen=True)
class ExactCompiledProblem:
    problem: CanonicalProblem
    groups: list[ExactGroup]
    fixed_assignment: dict[int, str]


def compile_to_exact(problem: CanonicalProblem) -> ExactCompiledProblem:
    """Compile a canonical problem into an exact-enumeration-friendly representation."""
    seen_indices: dict[int, str] = {}
    groups: list[ExactGroup] = []

    for constraint in problem.composition_constraints:
        if sum(constraint.counts.values()) != len(constraint.indices):
            raise ConfigError(
                f"Exact backend requires counts to sum to group size for {constraint.group_name!r}"
            )

        for idx in constraint.indices:
            if idx in seen_indices:
                prev = seen_indices[idx]
                raise NotImplementedError(f"Exact backend v1 requires disjoint groups; overlap at site {idx}: {prev}, {constraint.group_name}")
            seen_indices[idx] = constraint.group_name

            if idx not in problem.allowed_by_site:
                raise ConfigError(
                    f"Exact backend group {constraint.group_name!r} references site {idx} not present in allowed_by_site"
                )

        for species in constraint.counts:
            for idx in constraint.indices:
                if species not in problem.allowed_by_site[idx]:
                    raise ConfigError(
                        f"Species {species!r} in counts for group {constraint.group_name!r} is not allowed on site {idx}"
                    )

        groups.append(
            ExactGroup(
                name=constraint.group_name,
                indices=tuple(constraint.indices),
                counts=dict(constraint.counts),
            )
        )

    fixed_assignment = dict(problem.fixed_assignments)
    for idx, species in fixed_assignment.items():
        allowed = problem.allowed_by_site.get(idx)
        if allowed is None:
            raise ConfigError(f"Fixed assignment site {idx} not present in allowed_by_site")
        if species not in allowed:
            raise ConfigError(f"Fixed assignment species {species!r} is not allowed on site {idx}")

    return ExactCompiledProblem(problem=problem, groups=groups, fixed_assignment=fixed_assignment)
