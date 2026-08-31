"""Training lifecycle interfaces."""

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

__all__ = ["TrainResult", "Trainer"]


@dataclass(frozen=True, slots=True, kw_only=True)
class TrainResult:
    """Summary returned by a training run and stored with its manifest."""

    run_id: str
    steps: int
    checkpoint: str | None = None
    metrics: Mapping[str, float] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        require_non_empty(self.run_id, "run_id")
        require_non_negative_int(self.steps, "steps")
        if self.checkpoint is not None:
            require_non_empty(self.checkpoint, "checkpoint")
        frozen_metrics = freeze_value(self.metrics)
        if not isinstance(frozen_metrics, Mapping):
            raise ValueError("metrics must be a mapping")
        for name, value in frozen_metrics.items():
            require_finite_number(value, f"metrics.{name}")
        object.__setattr__(self, "metrics", frozen_metrics)
        object.__setattr__(self, "metadata", freeze_value(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""
        return {
            "run_id": self.run_id,
            "steps": self.steps,
            "checkpoint": self.checkpoint,
            "metrics": thaw_value(self.metrics),
            "metadata": thaw_value(self.metadata),
        }


@runtime_checkable
class Trainer(Protocol):
    """Framework-neutral training contract implemented by integrations."""

    def train(self) -> TrainResult:
        """Run training and return its provenance-friendly summary."""
