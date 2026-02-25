"""Command-line entrypoint placeholder.

TODO: wire `optimat run <input.yml>` to compilation/solve/decode pipeline.
"""

from __future__ import annotations

import argparse
from typing import Sequence


def build_parser() -> argparse.ArgumentParser:
    """Build the placeholder CLI parser."""
    parser = argparse.ArgumentParser(prog="optimat")
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Run an optimization from a YAML spec")
    run_parser.add_argument("input", help="Path to input YAML file")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint placeholder."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        raise NotImplementedError("CLI run flow is not implemented yet")

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
