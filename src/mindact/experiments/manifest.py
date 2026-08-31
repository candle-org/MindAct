"""Experiment provenance records.

Manifests are deliberately plain JSON-compatible data so they can be inspected,
compared and uploaded alongside checkpoints without requiring a tracking service.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = ["ExperimentManifest"]


@dataclass(frozen=True, slots=True)
class ExperimentManifest:
    """Identity and lineage for one train/evaluate run."""

    run_id: str
    experiment_name: str
    seed: int
    config: Mapping[str, Any]
    code_revision: str | None = None
    dataset: Mapping[str, Any] = field(default_factory=dict)
    policy: Mapping[str, Any] = field(default_factory=dict)
    environment: Mapping[str, Any] = field(default_factory=dict)
    checkpoint: Mapping[str, Any] | None = None
    evaluation: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        for name in ("run_id", "experiment_name"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        object.__setattr__(self, "config", dict(self.config))
        for name in ("dataset", "policy", "environment"):
            object.__setattr__(self, name, dict(getattr(self, name)))
        for name in ("checkpoint", "evaluation"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, dict(value))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""
        return {
            "run_id": self.run_id,
            "experiment_name": self.experiment_name,
            "seed": self.seed,
            "code_revision": self.code_revision,
            "config": dict(self.config),
            "dataset": dict(self.dataset),
            "policy": dict(self.policy),
            "environment": dict(self.environment),
            "checkpoint": None if self.checkpoint is None else dict(self.checkpoint),
            "evaluation": None if self.evaluation is None else dict(self.evaluation),
        }

    def write_json(self, path: str | Path) -> None:
        """Write the manifest atomically enough for ordinary local runs."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def read_json(cls, path: str | Path) -> ExperimentManifest:
        """Read a manifest previously written by :meth:`write_json`."""
        source = Path(path)
        try:
            data = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid experiment manifest '{source}': {exc}") from exc
        if not isinstance(data, dict):
            raise ValueError("experiment manifest must contain a JSON object")
        try:
            return cls(**data)
        except TypeError as exc:
            raise ValueError(f"invalid experiment manifest fields: {exc}") from exc
