"""Import smoke tests for the scaffold package."""

import optimat
import optimat.compile.bqm
import optimat.compile.cpsat
import optimat.config
import optimat.io.yaml
import optimat.spec.problem


def test_imports_smoke() -> None:
    assert optimat.__version__ == "0.1.0"
