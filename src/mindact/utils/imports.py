"""Optional dependency handling.

MindAct sits on top of heavyweight ecosystems (PyTorch, LeRobot, LIBERO). None of
them are imported at package import time: `import mindact` must work in a bare
environment so that configs, manifests and the CLI stay usable.
"""

from __future__ import annotations

import importlib
import importlib.util
from types import ModuleType

__all__ = [
    "MissingOptionalDependency",
    "is_available",
    "require_module",
]

# Module name -> extra that installs it.
_EXTRAS: dict[str, str] = {
    "torch": "torch",
    "huggingface_hub": "hub",
    "lerobot": "lerobot",
    "libero": "libero",
}


class MissingOptionalDependency(ImportError):
    """Raised when a feature is used without its optional dependency installed."""


def _install_hint(module: str) -> str:
    extra = _EXTRAS.get(module)
    if extra is None:
        return f"pip install {module}"
    return f'pip install "mindact[{extra}]"'


def is_available(module: str) -> bool:
    """Return whether ``module`` can be imported without importing it."""
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def require_module(module: str, *, feature: str | None = None) -> ModuleType:
    """Import ``module`` or raise an actionable error.

    Args:
        module: Top-level module name, e.g. ``"lerobot"``.
        feature: Human readable name of the MindAct feature requesting it.
    """
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        what = feature or f"the {module} integration"
        raise MissingOptionalDependency(
            f"{what} requires the '{module}' package, which is not installed. "
            f"Install it with: {_install_hint(module)}"
        ) from exc
