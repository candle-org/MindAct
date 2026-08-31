"""Local artifact paths for experiment outputs."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = ["ArtifactStore"]


def _safe_component(value: str, field_name: str) -> None:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a single non-empty path component")
    component = Path(value)
    if (
        not value.strip()
        or value in {".", ".."}
        or component.name != value
        or "/" in value
        or "\\" in value
    ):
        raise ValueError(f"{field_name} must be a single non-empty path component")


def _write_exclusive(path: str | Path, content: str) -> None:
    """Write text once, using a same-directory exclusive link."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f"artifact already exists: '{target}'")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent, text=True)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError as exc:
            raise FileExistsError(f"artifact already exists: '{target}'") from exc
        finally:
            Path(temporary).unlink(missing_ok=True)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


@dataclass(frozen=True, slots=True)
class ArtifactStore:
    """Conventional, inspectable paths associated with one run."""

    root: Path
    run_id: str

    def __post_init__(self) -> None:
        _safe_component(self.run_id, "run_id")
        object.__setattr__(self, "root", Path(self.root))

    @property
    def run_dir(self) -> Path:
        """Directory containing all artifacts for this run."""
        return self.root / self.run_id

    @property
    def manifest_path(self) -> Path:
        """Path to the run provenance manifest."""
        return self.run_dir / "manifest.json"

    @property
    def config_path(self) -> Path:
        """Path to the immutable configuration copy used by the run."""
        return self.run_dir / "config.yaml"

    @property
    def checkpoint_dir(self) -> Path:
        """Directory for checkpoints produced by training."""
        return self.run_dir / "checkpoints"

    @property
    def log_dir(self) -> Path:
        """Directory for structured logs and tracking outputs."""
        return self.run_dir / "logs"

    @property
    def evaluation_dir(self) -> Path:
        """Directory for aggregate results and rollout records."""
        return self.run_dir / "evaluation"

    @property
    def results_path(self) -> Path:
        """Path to aggregate evaluation metrics."""
        return self.evaluation_dir / "results.json"

    @property
    def episodes_path(self) -> Path:
        """Path to newline-delimited per-episode results."""
        return self.evaluation_dir / "episodes.jsonl"

    def create(self) -> ArtifactStore:
        """Create the standard run directories and return this store."""
        for directory in (self.run_dir, self.checkpoint_dir, self.log_dir, self.evaluation_dir):
            directory.mkdir(parents=True, exist_ok=True)
        return self

    @staticmethod
    def write_text(path: str | Path, content: str) -> None:
        """Write a text artifact without silently replacing an existing file."""
        _write_exclusive(path, content)

    @staticmethod
    def write_json(path: str | Path, value: dict[str, Any]) -> None:
        """Write a JSON artifact without silently replacing an existing file."""
        _write_exclusive(path, json.dumps(value, indent=2, sort_keys=True) + "\n")

    @staticmethod
    def write_jsonl(path: str | Path, values: list[dict[str, Any]]) -> None:
        """Write a newline-delimited JSON artifact without replacing it."""
        content = "".join(json.dumps(value, sort_keys=True) + "\n" for value in values)
        _write_exclusive(path, content)
