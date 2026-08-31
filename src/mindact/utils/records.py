"""Shared validation helpers for immutable, JSON-compatible result records.

Result objects cross the boundary between MindAct and external integrations,
then land in artifact files. Sharing one freeze/validate implementation keeps
``TrainResult``, ``EpisodeResult``, ``EpisodeRecord`` and ``EvaluationResult``
consistent about what can be persisted.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Real
from types import MappingProxyType
from typing import Any

__all__ = [
    "freeze_value",
    "require_finite_number",
    "require_non_empty",
    "require_non_negative_int",
    "thaw_value",
]


def freeze_value(value: Any) -> Any:
    """Return a deeply immutable, JSON-compatible copy of ``value``.

    Parameters
    ----------
    value:
        Mapping, sequence, string, boolean, integer, finite float or ``None``.

    Returns
    -------
    Any
        ``MappingProxyType`` for mappings, ``tuple`` for sequences, otherwise
        the original scalar.

    Raises
    ------
    ValueError
        If a mapping key is not a string, or a value is not a finite
        JSON-compatible scalar.
    """
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("result mapping keys must be strings")
        return MappingProxyType({key: freeze_value(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(freeze_value(item) for item in value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise ValueError("result values must be finite JSON-compatible values")


def thaw_value(value: Any) -> Any:
    """Return a mutable ``dict``/``list`` copy suitable for JSON serialization."""
    if isinstance(value, Mapping):
        return {key: thaw_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [thaw_value(item) for item in value]
    return value


def require_non_empty(value: str, field_name: str) -> None:
    """Raise ``ValueError`` unless ``value`` is a non-blank string."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


def require_finite_number(value: Real, field_name: str) -> None:
    """Raise ``ValueError`` unless ``value`` is a finite, non-boolean number."""
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{field_name} must be a finite number")


def require_non_negative_int(value: int, field_name: str) -> None:
    """Raise ``ValueError`` unless ``value`` is a non-negative, non-boolean integer."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
