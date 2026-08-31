"""MindAct command-line interface."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from mindact import __version__
from mindact.configs import ConfigError, ExperimentConfig
from mindact.evaluation import EvaluationRunner
from mindact.evaluation.fake import FakeEnvironment, FakePolicy


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

    if args.command == "eval":
        try:
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
