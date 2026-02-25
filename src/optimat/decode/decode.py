"""Sample decoding placeholder.

TODO: decode solver outputs using VariableMap metadata.
"""

from __future__ import annotations

from typing import Any

from optimat.mapping.varmap import VariableMap


def decode(sample: Any, varmap: VariableMap) -> dict[str, Any]:
    """Placeholder decoder entrypoint."""
    raise NotImplementedError("decode is not implemented yet")
