"""Experiment provenance records."""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

__all__ = ["ExperimentManifest"]


def _freeze(value: Any) -> Any:
    """Take a recursively immutable JSON-compatible snapshot."""
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("manifest mapping keys must be strings")
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    if isinstance(value, float):
        raise ValueError("manifest numeric values must be finite")
    raise ValueError("manifest values must be JSON-compatible")


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


@dataclass(frozen=True, slots=True, kw_only=True)
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
    schema_version: int = 1

    def __post_init__(self) -> None:
        for name in ("run_id", "experiment_name"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if (
            isinstance(self.schema_version, bool)
            or not isinstance(self.schema_version, int)
            or self.schema_version < 1
        ):
            raise ValueError("schema_version must be a positive integer")
        if self.code_revision is not None and (
            not isinstance(self.code_revision, str) or not self.code_revision.strip()
        ):
            raise ValueError("code_revision must be a non-empty string or None")
        object.__setattr__(self, "config", _freeze(self.config))
        for name in ("dataset", "policy", "environment"):
            object.__setattr__(self, name, _freeze(getattr(self, name)))
        for name in ("checkpoint", "evaluation"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _freeze(value))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "experiment_name": self.experiment_name,
            "seed": self.seed,
            "code_revision": self.code_revision,
            "config": _thaw(self.config),
            "dataset": _thaw(self.dataset),
            "policy": _thaw(self.policy),
            "environment": _thaw(self.environment),
            "checkpoint": None if self.checkpoint is None else _thaw(self.checkpoint),
            "evaluation": None if self.evaluation is None else _thaw(self.evaluation),
        }

    def write_json(self, path: str | Path) -> None:
        """Write this manifest once, without overwriting an existing file."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise FileExistsError(f"manifest already exists: '{target}'")
        if target.exists():
            raise FileExistsError(f"manifest already exists: '{target}'")
        payload = json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"
        descriptor, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent, text=True)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            try:
                os.link(temporary, target)
            except FileExistsError as exc:
                raise FileExistsError(f"manifest already exists: '{target}'") from exc
            finally:
                Path(temporary).unlink(missing_ok=True)
            directory_fd = os.open(target.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except BaseException:
            Path(temporary).unlink(missing_ok=True)
            raise

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
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid experiment manifest fields: {exc}") from exc
