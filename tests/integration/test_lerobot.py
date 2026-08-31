from __future__ import annotations

import pytest

from mindact.integrations.lerobot import load_lerobot
from mindact.utils.imports import is_available

pytestmark = pytest.mark.lerobot


@pytest.mark.skipif(not is_available("lerobot"), reason="lerobot is not installed")
def test_lerobot_integration_loads_installed_package() -> None:
    module = load_lerobot()

    assert module.__name__ == "lerobot"
