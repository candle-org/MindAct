from __future__ import annotations

from pathlib import Path

import pytest

from mindact import ExperimentManifest


def test_manifest_is_write_once_and_nested_values_are_snapshotted(tmp_path: Path) -> None:
    config = {"training": {"steps": 1}}
    manifest = ExperimentManifest(
        run_id="run",
        experiment_name="demo",
        seed=1,
        config=config,
    )
    config["training"]["steps"] = 2
    path = tmp_path / "manifest.json"

    manifest.write_json(path)
    original = path.read_text()

    with pytest.raises(FileExistsError, match="manifest already exists"):
        manifest.write_json(path)

    assert path.read_text() == original
    assert manifest.to_dict()["config"] == {"training": {"steps": 1}}
