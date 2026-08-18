"""Command-line interface for running RetinaFE experiments."""

from __future__ import annotations

import argparse
from pathlib import Path
import json

from .experiments import run_channel_validation, run_demo, run_robustness_sweeps


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser and its experiment subcommands.

    What happens in this function:
    1. A top-level parser is created for the ``retinafe`` command.
    2. Demo, validation, sweep, and all-in-one subcommands are registered.
    3. Shared config and output arguments are defined consistently.
    4. The parser is returned for use by ``main`` and tests.
    """

    parser = argparse.ArgumentParser(
        prog="retinafe",
        description="Frequency-encoded two-photon retinal imaging simulation",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command, help_text in (
        ("demo", "run the main retinal reconstruction comparison"),
        ("validate", "compare analytic and direct lock-in channel predictions"),
        ("sweeps", "run robustness and failure-regime sweeps"),
        ("all", "run demo, validation, and sweeps"),
    ):
        subparser = subparsers.add_parser(command, help=help_text)
        subparser.add_argument("--config", required=True, type=Path)
        subparser.add_argument("--output", required=True, type=Path)
        subparser.add_argument("--project-root", type=Path, default=None)
        if command in {"sweeps", "all"}:
            subparser.add_argument("--replicates", type=int, default=6)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Execute the selected RetinaFE experiment from command-line arguments.

    What happens in this function:
    1. Arguments are parsed and the output directory is selected.
    2. The requested experiment function is called.
    3. ``all`` writes each experiment into a dedicated subdirectory.
    4. A JSON summary is printed for scripts and continuous integration.
    5. Zero is returned on successful completion.
    """

    parser = build_parser()
    arguments = parser.parse_args(argv)
    if arguments.command == "demo":
        result = run_demo(
            arguments.config,
            arguments.output,
            arguments.project_root,
        )
    elif arguments.command == "validate":
        result = run_channel_validation(
            arguments.config,
            arguments.output,
            arguments.project_root,
        )
    elif arguments.command == "sweeps":
        result = run_robustness_sweeps(
            arguments.config,
            arguments.output,
            arguments.project_root,
            arguments.replicates,
        )
    else:
        result = {
            "demo": run_demo(
                arguments.config,
                arguments.output / "demo",
                arguments.project_root,
            ),
            "validation": run_channel_validation(
                arguments.config,
                arguments.output / "validation",
                arguments.project_root,
            ),
            "sweeps": run_robustness_sweeps(
                arguments.config,
                arguments.output / "sweeps",
                arguments.project_root,
                arguments.replicates,
            ),
        }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
