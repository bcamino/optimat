"""Declarative constraint specification placeholders.

TODO: define typed constraint spec hierarchy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ConstraintSpec:
    """Placeholder constraint specification."""

    kind: str = ""
    params: dict[str, Any] = field(default_factory=dict)
