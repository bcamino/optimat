"""Core problem specification dataclasses.

TODO: define validated schemas for problem inputs and runtime config.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class SiteSetSpec:
    """Placeholder site set definition."""

    data: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EncodingSpec:
    """Placeholder encoding configuration."""

    data: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ProblemSpec:
    """Placeholder top-level problem specification."""

    name: str = ""
    sites: SiteSetSpec = field(default_factory=SiteSetSpec)
    encoding: EncodingSpec = field(default_factory=EncodingSpec)
    constraints: list[Any] = field(default_factory=list)
    objective: list[Any] = field(default_factory=list)


@dataclass(slots=True)
class RunConfig:
    """Placeholder runtime/solver configuration."""

    solver: str = ""
    options: dict[str, Any] = field(default_factory=dict)
