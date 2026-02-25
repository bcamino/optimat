"""BQM compilation placeholder.

TODO: compile ProblemSpec into a backend-neutral or D-Wave BQM object.
"""

from __future__ import annotations

from typing import Any

from optimat.spec.problem import ProblemSpec, RunConfig


def compile_to_bqm(spec: ProblemSpec, run_config: RunConfig | None = None, **kwargs: Any) -> Any:
    """Placeholder BQM compiler entrypoint."""
    raise NotImplementedError("compile_to_bqm is not implemented yet")
