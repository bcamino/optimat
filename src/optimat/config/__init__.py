"""Public config loading and validation API.

TODO: expose versioned config parsers as the input schema evolves.
"""

from .io import read_yaml
from .schema import OptimatConfig
from .validate import ConfigError

__all__ = ["OptimatConfig", "ConfigError", "read_yaml"]
