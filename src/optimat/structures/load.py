"""Structure loading helpers backed by pymatgen."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pymatgen.core import Structure


def load_structure(path: str | Path) -> "Structure":
    """Load a structure file into a pymatgen Structure."""
    try:
        from pymatgen.core import Structure
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("pymatgen is required for structure loading") from exc

    return Structure.from_file(str(Path(path)))
