"""Environment interfaces for simulator integrations."""

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

__all__ = ["EpisodeResult", "EnvironmentAdapter"]


@dataclass(frozen=True, slots=True, kw_only=True)
class EpisodeResult:
    """Outcome and metadata for one deterministic rollout."""

    episode_id: str
    success: bool
    reward: float
    steps: int
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        require_non_empty(self.episode_id, "episode_id")
        if not isinstance(self.success, bool):
            raise ValueError("success must be a boolean")
        require_finite_number(self.reward, "reward")
        require_non_negative_int(self.steps, "steps")
        object.__setattr__(self, "metadata", freeze_value(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""
        return {
            "episode_id": self.episode_id,
            "success": self.success,
            "reward": self.reward,
            "steps": self.steps,
            "metadata": thaw_value(self.metadata),
        }


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
