"""Policy interfaces built around official PyTorch modules."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

__all__ = ["PolicyAdapter"]


@runtime_checkable
class PolicyAdapter(Protocol):
    """Inference and checkpoint boundary for an embodied policy."""

    @property
    def name(self) -> str:
        """Return the policy family name, such as ``act``."""

    @property
    def revision(self) -> str | None:
        """Return the model or Hub revision when known."""

    def predict(self, observation: Mapping[str, Any]) -> Any:
        """Predict an environment action from one observation."""

    def load(self, checkpoint: str) -> None:
        """Load weights from a local path or an upstream model reference."""
