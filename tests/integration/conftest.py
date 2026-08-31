"""Shared fixtures for optional-dependency integration tests."""

from __future__ import annotations

import pytest

from mindact.utils.imports import is_available


@pytest.fixture(scope="session")
def torch_module():
    """Return the imported torch module, skipping when torch is absent."""
    if not is_available("torch"):
        pytest.skip("torch is not installed; install with: pip install \"mindact[torch]\"")
    import torch

    return torch
