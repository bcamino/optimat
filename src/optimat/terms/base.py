"""Base term provider protocols and containers.

TODO: formalize the term-provider interface and canonical term objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


class TermProvider(Protocol):
    """Placeholder protocol for components that emit optimization terms."""

    def build_terms(self, *args: Any, **kwargs: Any) -> list["QuadraticTerm"]:
        """Return generated terms.

        TODO: define concrete input/output interfaces.
        """
        ...


@dataclass(slots=True)
class LinearTerm:
    """Placeholder linear term container."""

    var: str = ""
    bias: float = 0.0


@dataclass(slots=True)
class QuadraticTerm:
    """Placeholder quadratic term container."""

    u: str = ""
    v: str = ""
    bias: float = 0.0


@dataclass(slots=True)
class TermBundle:
    """Placeholder grouped term container."""

    linear: list[LinearTerm] = field(default_factory=list)
    quadratic: list[QuadraticTerm] = field(default_factory=list)
