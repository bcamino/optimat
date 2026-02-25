"""Generic pair-term loader/emitter placeholder.

TODO: support ingesting pair-list files and emitting term bundles.
"""

from __future__ import annotations

from pathlib import Path

from .base import TermBundle


def load_pair_terms(path: str | Path) -> TermBundle:
    """Load pair terms from a source file.

    TODO: implement parsing and validation.
    """
    raise NotImplementedError("load_pair_terms is not implemented yet")
