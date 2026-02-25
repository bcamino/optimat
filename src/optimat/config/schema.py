"""Configuration schema dataclasses for v1 YAML input.

TODO: expand schema coverage and introduce richer typing/validation in later versions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
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
    parameters_file: str | None = None
    species: dict[str, "SpeciesConfig"] = field(default_factory=dict)
    buckingham: "BuckinghamConfig | None" = None
    ewald: "EwaldConfig | None" = None


@dataclass(frozen=True)
class SpeciesConfig:
    charge: float


@dataclass(frozen=True)
class BuckinghamPairConfig:
    A: float
    rho: float
    C: float


@dataclass(frozen=True)
class BuckinghamSmoothingConfig:
    enabled: bool = False


@dataclass(frozen=True)
class BuckinghamConfig:
    cutoff: float
    smoothing: BuckinghamSmoothingConfig | None = None
    parameters: dict[str, BuckinghamPairConfig] = field(default_factory=dict)


@dataclass(frozen=True)
class EwaldConfig:
    engine: str
    mode: str
    accuracy: float | None = None
    real_space_cut: float | None = None
    recip_space_cut: float | None = None
    eta: float | None = None


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
