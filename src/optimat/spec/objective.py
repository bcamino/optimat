"""Objective and term specification placeholders.

TODO: define objective composition and weighting specs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ObjectiveTermSpec:
    """Placeholder objective term specification."""

    kind: str = ""
    weight: float = 1.0
    params: dict[str, Any] = field(default_factory=dict)
