"""YAML spec loader placeholder.

TODO: parse YAML into ProblemSpec and RunConfig.
"""

from __future__ import annotations

from pathlib import Path

from optimat.spec.problem import ProblemSpec, RunConfig


def load_yaml(path: str | Path) -> tuple[ProblemSpec, RunConfig]:
    """Load a problem spec and run config from YAML.

    TODO: implement YAML parsing/validation.
    """
    raise NotImplementedError("load_yaml is not implemented yet")
