"""Environment interfaces for simulator integrations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

__all__ = ["EpisodeResult", "EnvironmentAdapter"]


@dataclass(frozen=True, slots=True)
class EpisodeResult:
    """Outcome and metadata for one deterministic rollout."""

    episode_id: str
    success: bool
    reward: float
    steps: int
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.episode_id.strip():
            raise ValueError("episode_id must not be empty")
        if self.steps < 0:
            raise ValueError("steps must not be negative")
        object.__setattr__(self, "metadata", dict(self.metadata))


@runtime_checkable
class EnvironmentAdapter(Protocol):
    """Small simulator contract; LIBERO-specific details stay in integrations."""

    @property
    def name(self) -> str:
        """Return the environment family name."""

    @property
    def version(self) -> str | None:
        """Return the simulator or benchmark version when known."""

    def reset(self, *, seed: int | None = None) -> Mapping[str, Any]:
        """Reset one episode and return the initial observation."""

    def step(self, action: Any) -> tuple[Mapping[str, Any], float, bool, Mapping[str, Any]]:
        """Apply one action and return observation, reward, done and info."""

    def close(self) -> None:
        """Release simulator resources."""
