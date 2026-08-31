from __future__ import annotations

import pytest

from mindact.integrations.libero import load_libero
from mindact.utils.imports import is_available

pytestmark = pytest.mark.libero


@pytest.mark.skipif(not is_available("libero"), reason="libero is not installed")
def test_libero_integration_loads_installed_package() -> None:
    module = load_libero()

    assert module.__name__ == "libero"
