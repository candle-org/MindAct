"""Evaluation lifecycle interfaces and immutable result records."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from mindact.utils.records import (
    freeze_value,
    require_finite_number,
    require_non_empty,
    require_non_negative_int,
    thaw_value,
)

__all__ = ["EpisodeRecord", "EvaluationResult", "Evaluator"]


@dataclass(frozen=True, slots=True, kw_only=True)
class EpisodeRecord:
    """Serializable outcome for one evaluation episode."""

    episode_id: str
    seed: int
    success: bool
    reward: float
    steps: int
    truncated: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        require_non_empty(self.episode_id, "episode_id")
        require_non_negative_int(self.seed, "seed")
        if not isinstance(self.success, bool):
            raise ValueError("success must be a boolean")
        require_finite_number(self.reward, "reward")
        require_non_negative_int(self.steps, "steps")
        if not isinstance(self.truncated, bool):
            raise ValueError("truncated must be a boolean")
        object.__setattr__(self, "metadata", freeze_value(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""
        return {
            "episode_id": self.episode_id,
            "seed": self.seed,
            "success": self.success,
            "reward": self.reward,
            "steps": self.steps,
            "truncated": self.truncated,
            "metadata": thaw_value(self.metadata),
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class EvaluationResult:
    """Aggregate benchmark result associated with one checkpoint."""

    run_id: str
    episodes: int
    successes: int
    metrics: Mapping[str, float] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        require_non_empty(self.run_id, "run_id")
        require_non_negative_int(self.episodes, "episodes")
        if self.episodes == 0:
            raise ValueError("episodes must be greater than zero")
        require_non_negative_int(self.successes, "successes")
        if self.successes > self.episodes:
            raise ValueError("successes must be between zero and episodes")
        frozen_metrics = freeze_value(self.metrics)
        if not isinstance(frozen_metrics, Mapping):
            raise ValueError("metrics must be a mapping")
        for name, value in frozen_metrics.items():
            require_finite_number(value, f"metrics.{name}")
        object.__setattr__(self, "metrics", frozen_metrics)
        object.__setattr__(self, "metadata", freeze_value(self.metadata))

    @property
    def success_rate(self) -> float:
        """Return the fraction of successful episodes."""
        return self.successes / self.episodes

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""
        metrics = thaw_value(self.metrics)
        metrics.setdefault("success_rate", self.success_rate)
        return {
            "run_id": self.run_id,
            "episodes": self.episodes,
            "successes": self.successes,
            "success_rate": self.success_rate,
            "metrics": metrics,
            "metadata": thaw_value(self.metadata),
        }


@runtime_checkable
class Evaluator(Protocol):
    """Environment evaluation contract implemented by integrations."""

    def evaluate(self) -> EvaluationResult:
        """Run deterministic rollouts and return aggregate metrics."""
