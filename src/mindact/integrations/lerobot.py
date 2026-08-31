"""Optional LeRobot integration boundary."""

from __future__ import annotations

from types import ModuleType

from mindact.utils.imports import require_module

__all__ = ["load_lerobot"]


def load_lerobot() -> ModuleType:
    """Load LeRobot when a caller explicitly requests the integration."""
    return require_module("lerobot", feature="the LeRobot dataset or policy integration")
