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

    energy_parser = subparsers.add_parser("energy", help="Energy-model utilities")
    energy_subparsers = energy_parser.add_subparsers(dest="energy_command")
    energy_compile_parser = energy_subparsers.add_parser("compile", help="Compile energy terms from a YAML spec")
    energy_compile_parser.add_argument("--input", required=True, help="Path to input YAML file")

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

    if args.command == "energy" and args.energy_command == "compile":
        try:
            from optimat.energy import compile_energy_model

            config = read_yaml(args.input)
            terms = compile_energy_model(config)
        except (ConfigError, ModuleNotFoundError) as exc:
            print(f"ERROR: {exc}")
            return 1

        print(f"model sites: {terms.meta.get('model_site_count', 0)}")
        print(f"neighbor pairs within cutoff: {terms.meta.get('buckingham_neighbor_pairs', 0)}")
        print(f"pair entries: {len(terms.pair)}")
        for key, value in sorted(terms.pair.items())[:5]:
            print(f"sample {key}: {value:.6f}")
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
