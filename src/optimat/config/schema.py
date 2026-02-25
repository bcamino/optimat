"""Configuration schema dataclasses for v1 YAML input.

TODO: expand schema coverage and introduce richer typing/validation in later versions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class ProjectConfig:
    name: str
    output_dir: str


@dataclass(frozen=True)
class StructureConfig:
    file: str
    periodic: bool


@dataclass(frozen=True)
class CompositionCounts:
    mode: Literal["counts"]
    counts: dict[str, int]


@dataclass(frozen=True)
class SiteGroupConfig:
    name: str
    indices: list[int]
    allowed_species: list[str]
    composition: CompositionCounts


@dataclass(frozen=True)
class OccupancyConfig:
    site_groups: list[SiteGroupConfig]


@dataclass(frozen=True)
class EnergyModelConfig:
    type: str
    parameters_file: str


@dataclass(frozen=True)
class OptimisationConfig:
    objective: str


@dataclass(frozen=True)
class CpSatOptions:
    time_limit: int | float
    num_workers: int


@dataclass(frozen=True)
class SolverConfig:
    backend: str
    cp_sat: CpSatOptions | None = None


@dataclass(frozen=True)
class OptimatConfig:
    project: ProjectConfig
    structure: StructureConfig
    occupancy: OccupancyConfig
    energy_model: EnergyModelConfig
    optimisation: OptimisationConfig
    solver: SolverConfig
