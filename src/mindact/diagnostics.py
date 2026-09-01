"""Environment diagnostics for the MindAct command-line interface."""

from __future__ import annotations

import platform
import sys
from dataclasses import dataclass
from typing import Any

from mindact import __version__
from mindact.utils.imports import is_available

__all__ = ["DiagnosticCheck", "DoctorReport", "run_doctor"]


@dataclass(frozen=True, slots=True, kw_only=True)
class DiagnosticCheck:
    """One stable, human-readable environment diagnostic."""

    name: str
    status: str
    detail: str

    def __post_init__(self) -> None:
        for field_name in ("name", "status", "detail"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")

    def to_dict(self) -> dict[str, str]:
        """Return a JSON-compatible representation."""
        return {"name": self.name, "status": self.status, "detail": self.detail}


@dataclass(frozen=True, slots=True, kw_only=True)
class DoctorReport:
    """Immutable collection of checks produced by :func:`run_doctor`."""

    checks: tuple[DiagnosticCheck, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.checks, tuple) or not all(
            isinstance(check, DiagnosticCheck) for check in self.checks
        ):
            raise ValueError("checks must be a tuple of DiagnosticCheck values")

    @property
    def healthy(self) -> bool:
        """Return whether no diagnostic reports a core/runtime error.

        Missing optional integrations are expected in a minimal installation,
        while ``INFO`` entries describe scope rather than health.
        """
        return all(check.status != "ERROR" for check in self.checks)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""
        return {"healthy": self.healthy, "checks": [check.to_dict() for check in self.checks]}

    def render(self) -> str:
        """Render checks as stable, terminal-friendly text."""
        return "\n".join(
            f"{check.name:<18} {check.status:<12} {check.detail}" for check in self.checks
        )


_OPTIONAL_PACKAGES: tuple[tuple[str, str, str], ...] = (
    ("torch", "PyTorch", 'pip install "mindact[torch]"'),
    ("huggingface_hub", "Hugging Face Hub", 'pip install "mindact[hub]"'),
    ("lerobot", "LeRobot", 'pip install "mindact[lerobot]"'),
    ("libero", "LIBERO", 'pip install "mindact[libero]"'),
)


def run_doctor() -> DoctorReport:
    """Collect non-invasive diagnostics without importing optional packages.

    Package availability is reported separately from runtime validation. In
    particular, an ``OK`` package check does not claim that a simulator,
    renderer, device, or checkpoint is usable.
    """
    checks = [
        DiagnosticCheck(name="MindAct", status="OK", detail=__version__),
        DiagnosticCheck(name="Python", status="OK", detail=platform.python_version()),
        DiagnosticCheck(name="Platform", status="INFO", detail=sys.platform),
    ]
    for module_name, display_name, install_hint in _OPTIONAL_PACKAGES:
        if is_available(module_name):
            checks.append(
                DiagnosticCheck(name=display_name, status="FOUND", detail="importable module found")
            )
        else:
            checks.append(
                DiagnosticCheck(
                    name=display_name,
                    status="MISSING",
                    detail=f"not installed; install with {install_hint}",
                )
            )
    checks.append(
        DiagnosticCheck(
            name="Scope",
            status="INFO",
            detail="package checks only; renderer, device, and checkpoint access are not validated",
        )
    )
    return DoctorReport(checks=tuple(checks))
