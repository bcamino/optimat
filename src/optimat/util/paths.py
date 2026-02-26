"""Path utilities for config-relative file resolution."""

from __future__ import annotations

from pathlib import Path


def resolve_path(path: str, base_dir: Path) -> Path:
    """Resolve a possibly-relative path against the provided base directory."""
    p = Path(path)
    if p.is_absolute():
        return p
    return base_dir / p


def ensure_dir(path: Path) -> Path:
    """Create a directory (including parents) if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)
    return path
