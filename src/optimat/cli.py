"""Command-line entrypoint placeholder.

TODO: wire `optimat run <input.yml>` to compilation/solve/decode pipeline.
"""

from __future__ import annotations

import argparse
from typing import Sequence

from optimat.config import ConfigError, read_yaml


def build_parser() -> argparse.ArgumentParser:
    """Build the placeholder CLI parser."""
    parser = argparse.ArgumentParser(prog="optimat")
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Run an optimization from a YAML spec")
    run_parser.add_argument("input", help="Path to input YAML file")

    validate_parser = subparsers.add_parser("validate", help="Validate an optimat YAML config")
    validate_parser.add_argument("--input", required=True, help="Path to input YAML file")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint placeholder."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        raise NotImplementedError("CLI run flow is not implemented yet")

    if args.command == "validate":
        try:
            config = read_yaml(args.input)
        except ConfigError as exc:
            print(f"ERROR: {exc}")
            return 1
        print(f"OK: {config.project.name}")
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
