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

    solve_parser = subparsers.add_parser("solve", help="Solver backends")
    solve_subparsers = solve_parser.add_subparsers(dest="solve_command")
    solve_exact_parser = solve_subparsers.add_parser("exact", help="Run exact exhaustive solver")
    solve_exact_parser.add_argument("--input", required=True, help="Path to input YAML file")
    solve_exact_parser.add_argument("--top-k", type=int, default=1, help="Keep the top-k solutions")
    solve_exact_parser.add_argument("--max-evals", type=int, default=None, help="Optional evaluation limit")

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

            config, base_dir = read_yaml(args.input, return_base_dir=True)
            terms = compile_energy_model(config, base_dir=base_dir)
        except (ConfigError, ModuleNotFoundError) as exc:
            print(f"ERROR: {exc}")
            return 1

        print(f"model sites: {terms.meta.get('model_site_count', 0)}")
        print(f"neighbor pairs within cutoff: {terms.meta.get('buckingham_neighbor_pairs', 0)}")
        print(f"pair entries: {len(terms.pair)}")
        for key, value in sorted(terms.pair.items())[:5]:
            print(f"sample {key}: {value:.6f}")
        return 0

    if args.command == "solve" and args.solve_command == "exact":
        try:
            from optimat.backends.exact import compile_to_exact, run_exact
            from optimat.problem import build_problem

            config, base_dir = read_yaml(args.input, return_base_dir=True)
            problem = build_problem(config, base_dir=base_dir)
            compiled = compile_to_exact(problem)
            result = run_exact(compiled, max_evals=args.max_evals, top_k=args.top_k)
        except (ConfigError, ModuleNotFoundError, NotImplementedError, RuntimeError, ValueError) as exc:
            print(f"ERROR: {exc}")
            return 1

        print(f"best energy: {result.best_energy:.12f}")
        variable_assign = {i: result.best_assignment[i] for i in problem.variable_sites if i in result.best_assignment}
        print(f"best assignment (variable sites): {variable_assign}")
        print(
            f"evaluated: {result.stats.num_evaluated} "
            f"feasible: {result.stats.num_feasible} "
            f"time_sec: {result.stats.time_sec:.6f}"
        )
        if result.top_k:
            for idx, (energy, assignment) in enumerate(result.top_k, start=1):
                variable_top = {i: assignment[i] for i in problem.variable_sites if i in assignment}
                print(f"top[{idx}] energy={energy:.12f} assignment={variable_top}")
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
