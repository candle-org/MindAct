"""Typed experiment configuration for reproducible embodied-policy runs."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from numbers import Real
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml

__all__ = [
    "ConfigError",
    "DatasetConfig",
    "EnvironmentConfig",
    "EvaluationConfig",
    "ExperimentConfig",
    "PolicyConfig",
    "TrainingConfig",
]


class ConfigError(ValueError):
    """Raised when an experiment configuration is invalid."""


def _non_empty(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"'{field_name}' must be a non-empty string")


def _optional_string(value: str | None, field_name: str) -> None:
    if value is not None:
        _non_empty(value, field_name)


def _positive(value: Real, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value <= 0:
        raise ConfigError(f"'{field_name}' must be greater than zero")


def _positive_int(value: int, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigError(f"'{field_name}' must be greater than zero")


def _non_negative_int(value: int, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ConfigError(f"'{field_name}' must be a non-negative integer")


def _section(data: Mapping[str, Any], name: str) -> dict[str, Any]:
    value = data.get(name)
    if not isinstance(value, Mapping):
        raise ConfigError(f"'{name}' must be a mapping")
    return dict(value)


def _reject_unknown(data: Mapping[str, Any], allowed: set[str], section: str) -> None:
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ConfigError(f"unknown field(s) in '{section}': {', '.join(unknown)}")


def _freeze(value: Any, field_name: str) -> Any:
    """Return a recursively immutable, YAML/JSON-compatible snapshot."""
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ConfigError(f"'{field_name}' mapping keys must be strings")
        return MappingProxyType({key: _freeze(item, f"{field_name}.{key}") for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item, f"{field_name}[]") for item in value)
    if isinstance(value, (str, bool, int)) or value is None:
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise ConfigError(f"'{field_name}' contains a value that is not YAML/JSON compatible")


def _snapshot_options(value: Mapping[str, Any], field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"'{field_name}' must be a mapping")
    return _freeze(value, field_name)


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


@dataclass(frozen=True, slots=True, kw_only=True)
class DatasetConfig:
    """Dataset identity and loader-specific options."""

    repo_id: str
    revision: str | None = None
    split: str = "train"
    options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _non_empty(self.repo_id, "dataset.repo_id")
        _optional_string(self.revision, "dataset.revision")
        _non_empty(self.split, "dataset.split")
        object.__setattr__(self, "options", _snapshot_options(self.options, "dataset.options"))


@dataclass(frozen=True, slots=True, kw_only=True)
class PolicyConfig:
    """Policy architecture or checkpoint identity."""

    name: str
    pretrained_model: str | None = None
    revision: str | None = None
    options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _non_empty(self.name, "policy.name")
        _optional_string(self.pretrained_model, "policy.pretrained_model")
        _optional_string(self.revision, "policy.revision")
        object.__setattr__(self, "options", _snapshot_options(self.options, "policy.options"))


@dataclass(frozen=True, slots=True, kw_only=True)
class EnvironmentConfig:
    """Evaluation environment and deterministic task selection."""

    name: str
    task_suite: str | None = None
    task_ids: tuple[int, ...] = ()
    options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _non_empty(self.name, "environment.name")
        if not isinstance(self.task_ids, (list, tuple)):
            raise ConfigError("'environment.task_ids' must be a list")
        if any(
            isinstance(task_id, bool) or not isinstance(task_id, int) or task_id < 0
            for task_id in self.task_ids
        ):
            raise ConfigError("'environment.task_ids' must contain non-negative integers")
        _optional_string(self.task_suite, "environment.task_suite")
        object.__setattr__(self, "task_ids", tuple(self.task_ids))
        object.__setattr__(self, "options", _snapshot_options(self.options, "environment.options"))


@dataclass(frozen=True, slots=True, kw_only=True)
class TrainingConfig:
    """Framework-neutral training controls."""

    steps: int
    batch_size: int
    learning_rate: float
    log_every: int = 100
    checkpoint_every: int = 10_000
    options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("steps", "batch_size", "log_every", "checkpoint_every"):
            _positive_int(getattr(self, name), f"training.{name}")
        _positive(self.learning_rate, "training.learning_rate")
        object.__setattr__(self, "options", _snapshot_options(self.options, "training.options"))


@dataclass(frozen=True, slots=True, kw_only=True)
class EvaluationConfig:
    """Rollout and reporting controls."""

    episodes: int = 10
    max_steps: int | None = None
    record_video: bool = True
    options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _positive_int(self.episodes, "evaluation.episodes")
        if self.max_steps is not None:
            _positive_int(self.max_steps, "evaluation.max_steps")
        if not isinstance(self.record_video, bool):
            raise ConfigError("'evaluation.record_video' must be a boolean")
        object.__setattr__(self, "options", _snapshot_options(self.options, "evaluation.options"))


@dataclass(frozen=True, slots=True, kw_only=True)
class ExperimentConfig:
    """Complete configuration for one train-and-evaluate experiment."""

    name: str
    dataset: DatasetConfig
    policy: PolicyConfig
    environment: EnvironmentConfig
    training: TrainingConfig
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    seed: int = 0
    output_dir: Path = Path("outputs")

    def __post_init__(self) -> None:
        _non_empty(self.name, "name")
        _non_negative_int(self.seed, "seed")
        if not isinstance(self.dataset, DatasetConfig):
            raise ConfigError("'dataset' must be a DatasetConfig")
        if not isinstance(self.policy, PolicyConfig):
            raise ConfigError("'policy' must be a PolicyConfig")
        if not isinstance(self.environment, EnvironmentConfig):
            raise ConfigError("'environment' must be an EnvironmentConfig")
        if not isinstance(self.training, TrainingConfig):
            raise ConfigError("'training' must be a TrainingConfig")
        if not isinstance(self.evaluation, EvaluationConfig):
            raise ConfigError("'evaluation' must be an EvaluationConfig")
        if not isinstance(self.output_dir, (str, Path)) or not str(self.output_dir).strip():
            raise ConfigError("'output_dir' must not be empty")
        try:
            output_dir = Path(self.output_dir)
        except TypeError as exc:
            raise ConfigError("'output_dir' must be a path-like value") from exc
        object.__setattr__(self, "output_dir", output_dir)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ExperimentConfig:
        """Build a strict configuration from a mapping."""
        if not isinstance(data, Mapping):
            raise ConfigError("experiment config must be a mapping")
        _reject_unknown(
            data,
            {"name", "seed", "output_dir", "dataset", "policy", "environment", "training", "evaluation"},
            "root",
        )

        dataset = _section(data, "dataset")
        policy = _section(data, "policy")
        environment = _section(data, "environment")
        training = _section(data, "training")
        evaluation_data = data.get("evaluation", {})
        if not isinstance(evaluation_data, Mapping):
            raise ConfigError("'evaluation' must be a mapping")
        evaluation = dict(evaluation_data)
        sections = (
            (dataset, {"repo_id", "revision", "split", "options"}, "dataset"),
            (policy, {"name", "pretrained_model", "revision", "options"}, "policy"),
            (environment, {"name", "task_suite", "task_ids", "options"}, "environment"),
            (
                training,
                {"steps", "batch_size", "learning_rate", "log_every", "checkpoint_every", "options"},
                "training",
            ),
            (evaluation, {"episodes", "max_steps", "record_video", "options"}, "evaluation"),
        )
        for section_data, allowed, section_name in sections:
            _reject_unknown(section_data, allowed, section_name)

        task_ids = environment.get("task_ids", ())
        if not isinstance(task_ids, (list, tuple)):
            raise ConfigError("'environment.task_ids' must be a list")
        environment["task_ids"] = tuple(task_ids)

        try:
            return cls(
                name=data["name"],
                seed=data.get("seed", 0),
                output_dir=data.get("output_dir", "outputs"),
                dataset=DatasetConfig(**dataset),
                policy=PolicyConfig(**policy),
                environment=EnvironmentConfig(**environment),
                training=TrainingConfig(**training),
                evaluation=EvaluationConfig(**evaluation),
            )
        except KeyError as exc:
            raise ConfigError(f"missing required field: {exc.args[0]}") from exc
        except TypeError as exc:
            raise ConfigError(str(exc)) from exc

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        """Load a configuration from a YAML file."""
        config_path = Path(path)
        try:
            data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        except OSError as exc:
            raise ConfigError(f"cannot read config '{config_path}': {exc}") from exc
        except yaml.YAMLError as exc:
            raise ConfigError(f"invalid YAML in '{config_path}': {exc}") from exc
        return cls.from_dict(data)

    def to_dict(self) -> dict[str, Any]:
        """Return a stable, YAML- and JSON-safe representation."""
        return {
            "name": self.name,
            "seed": self.seed,
            "output_dir": str(self.output_dir),
            "dataset": {
                "repo_id": self.dataset.repo_id,
                "revision": self.dataset.revision,
                "split": self.dataset.split,
                "options": _thaw(self.dataset.options),
            },
            "policy": {
                "name": self.policy.name,
                "pretrained_model": self.policy.pretrained_model,
                "revision": self.policy.revision,
                "options": _thaw(self.policy.options),
            },
            "environment": {
                "name": self.environment.name,
                "task_suite": self.environment.task_suite,
                "task_ids": list(self.environment.task_ids),
                "options": _thaw(self.environment.options),
            },
            "training": {
                "steps": self.training.steps,
                "batch_size": self.training.batch_size,
                "learning_rate": self.training.learning_rate,
                "log_every": self.training.log_every,
                "checkpoint_every": self.training.checkpoint_every,
                "options": _thaw(self.training.options),
            },
            "evaluation": {
                "episodes": self.evaluation.episodes,
                "max_steps": self.evaluation.max_steps,
                "record_video": self.evaluation.record_video,
                "options": _thaw(self.evaluation.options),
            },
        }

    def to_yaml(self, path: str | Path) -> None:
        """Write this configuration as portable YAML."""
        config_path = Path(path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(yaml.safe_dump(self.to_dict(), sort_keys=False), encoding="utf-8")
