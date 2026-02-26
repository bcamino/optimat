"""Exact exhaustive enumeration solver."""

from __future__ import annotations

import heapq
import time
from itertools import combinations

from optimat.backends.exact.compile import ExactCompiledProblem, ExactGroup
from optimat.results import SolveResult, SolveStats


class _StopEnumeration(Exception):
    """Internal control-flow exception for max_evals early stop."""


def run_exact(
    compiled: ExactCompiledProblem,
    *,
    max_evals: int | None = None,
    top_k: int = 1,
) -> SolveResult:
    """Enumerate all feasible assignments (or until max_evals) and return the best result."""
    if top_k < 1:
        raise ValueError("top_k must be >= 1")

    start = time.perf_counter()
    problem = compiled.problem

    base_assignment = dict(compiled.fixed_assignment)
    best_energy = float("inf")
    best_assignment: dict[int, str] | None = None

    num_feasible = 0
    num_evaluated = 0
    tie_counter = 0
    top_heap: list[tuple[float, int, dict[int, str]]] = []

    def maybe_record_top(energy: float, assignment: dict[int, str]) -> None:
        nonlocal tie_counter
        if top_k <= 1:
            return
        item = (-energy, tie_counter, dict(assignment))
        tie_counter += 1
        if len(top_heap) < top_k:
            heapq.heappush(top_heap, item)
            return
        worst_energy = -top_heap[0][0]
        if energy < worst_energy:
            heapq.heapreplace(top_heap, item)

    def evaluate_candidate(current_assignment: dict[int, str]) -> None:
        nonlocal best_energy, best_assignment, num_feasible, num_evaluated
        if max_evals is not None and num_evaluated >= max_evals:
            raise _StopEnumeration

        energy = problem.energy_terms.energy_of_assignment(current_assignment)
        num_feasible += 1
        num_evaluated += 1

        if energy < best_energy:
            best_energy = energy
            best_assignment = dict(current_assignment)
        maybe_record_top(energy, current_assignment)

    def dfs(group_idx: int, current_assignment: dict[int, str]) -> None:
        if group_idx >= len(compiled.groups):
            evaluate_candidate(current_assignment)
            return

        group = compiled.groups[group_idx]
        for group_assignment in generate_group_assignments(group):
            touched: list[int] = []
            conflict = False
            for idx, species in group_assignment.items():
                if idx in current_assignment:
                    if current_assignment[idx] != species:
                        conflict = True
                        break
                    continue
                current_assignment[idx] = species
                touched.append(idx)

            if not conflict:
                dfs(group_idx + 1, current_assignment)

            for idx in touched:
                del current_assignment[idx]

    try:
        dfs(0, base_assignment)
    except _StopEnumeration:
        pass

    elapsed = time.perf_counter() - start
    if best_assignment is None:
        raise RuntimeError("Exact solver evaluated no feasible assignments")

    top_out: list[tuple[float, dict[int, str]]] | None = None
    if top_k > 1:
        items = [(-neg_energy, assignment) for (neg_energy, _, assignment) in top_heap]
        top_out = sorted(items, key=lambda x: x[0])

    return SolveResult(
        best_energy=best_energy,
        best_assignment=best_assignment,
        stats=SolveStats(num_feasible=num_feasible, num_evaluated=num_evaluated, time_sec=elapsed),
        top_k=top_out,
    )


def generate_group_assignments(group: ExactGroup):
    """Yield all distinct per-site assignments satisfying group counts."""
    indices = tuple(group.indices)
    counts = {sp: int(c) for sp, c in group.counts.items() if int(c) > 0}

    if sum(counts.values()) != len(indices):
        raise ValueError(f"Invalid counts for group {group.name!r}: sum(counts) != len(indices)")

    species_items = list(counts.items())
    if not species_items:
        if not indices:
            yield {}
        return

    if len(species_items) == 1:
        species, _ = species_items[0]
        yield {i: species for i in indices}
        return

    if len(species_items) == 2:
        (s1, c1), (s2, c2) = species_items
        # Use smaller count for fewer combinations; swap mapping if needed.
        if c2 < c1:
            s1, s2 = s2, s1
            c1, c2 = c2, c1
        idx_set = set(indices)
        for chosen in combinations(indices, c1):
            chosen_set = set(chosen)
            assignment = {i: s1 for i in chosen}
            for i in idx_set - chosen_set:
                assignment[i] = s2
            yield assignment
        return

    species_names = [s for s, _ in species_items]
    species_counts = [c for _, c in species_items]

    def rec(remaining_indices: tuple[int, ...], pos: int):
        if pos == len(species_names) - 1:
            species = species_names[pos]
            count = species_counts[pos]
            if count != len(remaining_indices):
                return
            yield {i: species for i in remaining_indices}
            return

        species = species_names[pos]
        count = species_counts[pos]
        if count == 0:
            yield from rec(remaining_indices, pos + 1)
            return

        for chosen in combinations(remaining_indices, count):
            chosen_set = set(chosen)
            rest = tuple(i for i in remaining_indices if i not in chosen_set)
            for sub in rec(rest, pos + 1):
                assignment = {i: species for i in chosen}
                assignment.update(sub)
                yield assignment

    yield from rec(indices, 0)
