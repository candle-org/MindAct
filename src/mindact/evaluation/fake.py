"""Deterministic, dependency-free adapters for smoke tests and examples."""

from __future__ import annotations

import random
from collections.abc import Mapping
from typing import Any

__all__ = ["FakeEnvironment", "FakePolicy"]


class FakePolicy:
    """A deterministic policy that returns the observation's target action."""

    name = "fake"
    revision = "builtin"

    def __init__(self) -> None:
        self.loaded_checkpoint: str | None = None

    def predict(self, observation: Mapping[str, Any]) -> int:
        """Return the action encoded in the observation."""
        return int(observation.get("target_action", 0))

    def load(self, checkpoint: str) -> None:
        """Record a checkpoint reference without reading its contents."""
        self.loaded_checkpoint = checkpoint


class FakeEnvironment:
    """A tiny seeded environment that completes after one correct action."""

    name = "fake"
    version = "1"

    def __init__(self) -> None:
        self._rng = random.Random()
        self._target_action = 0
        self._done = False

    def reset(self, *, seed: int | None = None) -> Mapping[str, Any]:
        """Reset using a local random generator seeded by the runner."""
        self._rng.seed(seed)
        self._target_action = self._rng.randrange(2)
        self._done = False
        return {"target_action": self._target_action, "seed": seed}

    def step(self, action: Any) -> tuple[Mapping[str, Any], float, bool, Mapping[str, Any]]:
        """Complete the episode and report whether the action was correct."""
        if self._done:
            raise RuntimeError("fake environment cannot step after done")
        self._done = True
        success = action == self._target_action
        return {}, 1.0 if success else 0.0, True, {"success": success}

    def close(self) -> None:
        """Release the environment (there are no external resources)."""
        self._done = True
