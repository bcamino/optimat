"""CP-SAT compilation placeholder.

TODO: compile ProblemSpec into an OR-Tools CP-SAT model.
"""

from __future__ import annotations

from typing import Any

from optimat.spec.problem import ProblemSpec, RunConfig


def compile_to_cpsat(spec: ProblemSpec, run_config: RunConfig | None = None, **kwargs: Any) -> Any:
    """Placeholder CP-SAT compiler entrypoint."""
    raise NotImplementedError("compile_to_cpsat is not implemented yet")
