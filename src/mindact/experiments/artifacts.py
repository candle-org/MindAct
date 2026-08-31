"""Local artifact paths for experiment outputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

__all__ = ["ArtifactStore"]


@dataclass(frozen=True, slots=True)
class ArtifactStore:
    """Conventional, inspectable paths associated with one run."""

    root: Path
    run_id: str

    def __post_init__(self) -> None:
        run_component = Path(self.run_id)
        if (
            not self.run_id.strip()
            or self.run_id in {".", ".."}
            or run_component.name != self.run_id
            or "/" in self.run_id
            or "\\" in self.run_id
        ):
            raise ValueError("run_id must be a single non-empty path component")
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

    def create(self) -> ArtifactStore:
        """Create the standard run directories and return this store."""
        for directory in (self.run_dir, self.checkpoint_dir, self.log_dir, self.evaluation_dir):
            directory.mkdir(parents=True, exist_ok=True)
        return self
