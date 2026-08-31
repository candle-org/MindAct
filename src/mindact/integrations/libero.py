"""Optional LIBERO integration boundary.

The upstream LIBERO benchmark is distributed primarily through its GitHub
repository and environment setup, not as a universally compatible wheel. MindAct
therefore checks for an installation at runtime instead of pinning a fake PyPI
requirement into the base package.
"""

from __future__ import annotations

from types import ModuleType

from mindact.utils.imports import require_module

__all__ = ["load_libero"]


def load_libero() -> ModuleType:
    """Load a user-installed LIBERO checkout or package."""
    return require_module("libero", feature="the LIBERO simulation integration")
