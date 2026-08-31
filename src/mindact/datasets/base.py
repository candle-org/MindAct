"""Dataset interfaces owned by MindAct.

Concrete storage and decoding remain in upstream projects such as LeRobot.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

__all__ = ["DatasetInfo", "DatasetAdapter"]


@dataclass(frozen=True, slots=True)
class DatasetInfo:
    """Stable identity and metadata captured in an experiment manifest."""

    repo_id: str
    revision: str | None = None
    split: str = "train"
    features: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.repo_id.strip():
            raise ValueError("repo_id must not be empty")
        if not self.split.strip():
            raise ValueError("split must not be empty")
        object.__setattr__(self, "features", dict(self.features))


@runtime_checkable
class DatasetAdapter(Protocol):
    """Minimal dataset contract required by MindAct training."""

    @property
    def info(self) -> DatasetInfo:
        """Return the dataset identity and schema."""

    def __len__(self) -> int:
        """Return the number of training examples or episodes."""

    def iter_batches(self, batch_size: int) -> Iterable[Mapping[str, Any]]:
        """Yield framework-native batches without prescribing their representation."""
