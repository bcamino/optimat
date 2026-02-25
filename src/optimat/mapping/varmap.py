"""Variable map placeholders.

TODO: implement stable variable indexing and reverse lookups.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class VariableMap:
    """Placeholder variable map container."""

    variables: dict[str, int] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add(self, name: str) -> int:
        """Register a variable name.

        TODO: implement deterministic indexing.
        """
        raise NotImplementedError("VariableMap.add is not implemented yet")
