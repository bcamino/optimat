"""JSON result serialization placeholders.

TODO: define stable result schemas and serialization helpers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def save_results(path: str | Path, results: dict[str, Any]) -> None:
    """Placeholder JSON result writer."""
    raise NotImplementedError("save_results is not implemented yet")


def load_results(path: str | Path) -> dict[str, Any]:
    """Placeholder JSON result reader."""
    raise NotImplementedError("load_results is not implemented yet")
