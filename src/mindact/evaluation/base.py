"""Evaluation lifecycle interfaces."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

__all__ = ["EvaluationResult", "Evaluator"]


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """Aggregate benchmark result associated with one checkpoint."""

    run_id: str
    episodes: int
    successes: int
    metrics: Mapping[str, float] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id must not be empty")
        if self.episodes <= 0:
            raise ValueError("episodes must be greater than zero")
        if self.successes < 0 or self.successes > self.episodes:
            raise ValueError("successes must be between zero and episodes")
        object.__setattr__(self, "metrics", dict(self.metrics))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def success_rate(self) -> float:
        """Return the fraction of successful episodes."""
        return self.successes / self.episodes


@runtime_checkable
class Evaluator(Protocol):
    """Environment evaluation contract implemented by integrations."""

    def evaluate(self) -> EvaluationResult:
        """Run deterministic rollouts and return aggregate metrics."""
