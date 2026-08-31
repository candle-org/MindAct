"""MindAct command-line interface."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from mindact import __version__
from mindact.configs import ConfigError, ExperimentConfig


def build_parser() -> argparse.ArgumentParser:
    """Build the public command-line parser."""
    parser = argparse.ArgumentParser(
        prog="mindact",
        description="Train and evaluate embodied policies with reproducible experiment records.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command")

    config_parser = subparsers.add_parser(
        "config-check",
        help="validate an experiment YAML file without starting a run",
    )
    config_parser.add_argument("path", help="path to an experiment YAML file")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the MindAct CLI and return a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "config-check":
        try:
            config = ExperimentConfig.from_yaml(args.path)
        except ConfigError as exc:
            parser.error(str(exc))
        print(f"valid: {config.name}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
