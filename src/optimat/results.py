"""Shared solve result dataclasses."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SolveStats:
    num_feasible: int
    num_evaluated: int
    time_sec: float


@dataclass(frozen=True)
class SolveResult:
    best_energy: float
    best_assignment: dict[int, str]
    stats: SolveStats
    top_k: list[tuple[float, dict[int, str]]] | None = None
