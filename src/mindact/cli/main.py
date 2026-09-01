"""MindAct command-line interface."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from mindact import __version__
from mindact.configs import ConfigError, ExperimentConfig
from mindact.diagnostics import run_doctor
from mindact.evaluation import EvaluationRunner
from mindact.experiments import ExperimentManifest


def _manifest_path(value: str) -> str:
    """Resolve a manifest file or run directory supplied on the CLI."""
    path = Path(value)
    if path.is_dir():
        path /= "manifest.json"
    return str(path)


def _print_manifest(path: str) -> int:
    """Print stable identity fields from an experiment manifest."""
    manifest = ExperimentManifest.read_json(path)
    data = manifest.to_dict()
    print(f"run_id: {data['run_id']}")
    print(f"experiment: {data['experiment_name']}")
    print(f"seed: {data['seed']}")
    print(f"code_revision: {data['code_revision'] or '-'}")
    for section in ("dataset", "policy", "environment", "checkpoint", "evaluation"):
        value = data[section]
        print(f"{section}: {json.dumps(value, sort_keys=True)}")
    return 0


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

    subparsers.add_parser(
        "doctor",
        help="check core and optional integration availability",
    )

    manifest_parser = subparsers.add_parser(
        "manifest",
        help="inspect experiment provenance",
    )
    manifest_subparsers = manifest_parser.add_subparsers(dest="manifest_command", required=True)
    show_parser = manifest_subparsers.add_parser("show", help="display a manifest")
    show_parser.add_argument("path", help="manifest.json or an experiment run directory")

    eval_parser = subparsers.add_parser(
        "eval",
        help="run the dependency-free fake evaluator",
    )
    eval_parser.add_argument("path", help="path to an experiment YAML file")
    eval_parser.add_argument("--runner", choices=("fake",), default="fake")
    eval_parser.add_argument("--run-id", default=None, help="stable run identifier for output artifacts")
    eval_parser.add_argument("--output-dir", default=None, help="override the configured output directory")
    eval_parser.add_argument("--episodes", type=int, default=None, help="override evaluation episode count")
    eval_parser.add_argument("--max-steps", type=int, default=None, help="override the evaluation step limit")
    eval_parser.add_argument("--seed", type=int, default=None, help="override the experiment seed")
    eval_parser.add_argument("--checkpoint", default=None, help="checkpoint reference passed to the policy")
    return parser


def _override_config(args: argparse.Namespace) -> ExperimentConfig:
    """Load a config and apply CLI overrides through a fresh validated instance."""
    config = ExperimentConfig.from_yaml(args.path)
    data = config.to_dict()
    if args.output_dir is not None:
        data["output_dir"] = args.output_dir
    if args.seed is not None:
        data["seed"] = args.seed
    if args.episodes is not None:
        data["evaluation"]["episodes"] = args.episodes
    if args.max_steps is not None:
        data["evaluation"]["max_steps"] = args.max_steps
    return ExperimentConfig.from_dict(data)


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

    if args.command == "doctor":
        print(run_doctor().render())
        return 0

    if args.command == "manifest" and args.manifest_command == "show":
        try:
            return _print_manifest(_manifest_path(args.path))
        except (OSError, TypeError, ValueError) as exc:
            parser.error(str(exc))

    if args.command == "eval":
        try:
            from mindact.evaluation.fake import FakeEnvironment, FakePolicy

            config = _override_config(args)
            policy = FakePolicy()
            runner = EvaluationRunner(
                config=config,
                policy=policy,
                environment_factory=FakeEnvironment,
                run_id=args.run_id,
                checkpoint=args.checkpoint,
            )
            result = runner.evaluate()
        except (ConfigError, FileExistsError, OSError, TypeError, ValueError) as exc:
            parser.error(str(exc))
        print(f"evaluated: {result.run_id} (success_rate={result.success_rate:.3f})")
        print(f"results: {runner.store.results_path}")
        return 0

    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
