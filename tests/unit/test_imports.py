from __future__ import annotations

import sys
import types

import pytest

from mindact.utils.imports import MissingOptionalDependency, is_available, require_module


def test_core_import_has_no_heavy_integration_imports() -> None:
    import mindact

    assert mindact.__version__ == "0.1.0"
    assert "lerobot" not in sys.modules
    assert "libero" not in sys.modules


def test_is_available_for_stdlib_module() -> None:
    assert is_available("json")


def test_require_module_returns_imported_module() -> None:
    assert require_module("json").dumps({"ok": True}) == '{"ok": true}'


def test_require_module_reports_install_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    module_name = "mindact_test_missing_optional"
    monkeypatch.setattr(
        "mindact.utils.imports.importlib.import_module",
        lambda name: (_ for _ in ()).throw(ModuleNotFoundError(name=module_name))
        if name == module_name
        else __import__(name),
    )

    error_pattern = r'pip install "mindact\[.*\]"|pip install mindact_test'
    with pytest.raises(MissingOptionalDependency, match=error_pattern):
        require_module(module_name, feature="test integration")


def test_require_module_can_load_a_runtime_module(monkeypatch: pytest.MonkeyPatch) -> None:
    module_name = "mindact_test_runtime_module"
    module = types.ModuleType(module_name)
    monkeypatch.setitem(sys.modules, module_name, module)
    monkeypatch.setattr(
        "mindact.utils.imports.importlib.util.find_spec",
        lambda name: object() if name == module_name else None,
    )

    assert require_module(module_name) is module
