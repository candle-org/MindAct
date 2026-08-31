"""Training lifecycle interfaces."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

__all__ = ["TrainResult", "Trainer"]


@dataclass(frozen=True, slots=True)
class TrainResult:
    """Summary returned by a training run and stored with its manifest."""

    run_id: str
    steps: int
    checkpoint: str | None = None
    metrics: Mapping[str, float] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id must not be empty")
        if self.steps < 0:
            raise ValueError("steps must not be negative")
        object.__setattr__(self, "metrics", dict(self.metrics))
        object.__setattr__(self, "metadata", dict(self.metadata))


@runtime_checkable
class Trainer(Protocol):
    """Framework-neutral training contract implemented by integrations."""

    def train(self) -> TrainResult:
        """Run training and return its provenance-friendly summary."""
