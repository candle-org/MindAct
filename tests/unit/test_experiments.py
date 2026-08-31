from __future__ import annotations

import json
from pathlib import Path

import pytest

from mindact import ArtifactStore, ExperimentManifest


def test_artifact_store_creates_standard_directories(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "outputs", "run-001").create()

    assert store.run_dir.is_dir()
    assert store.checkpoint_dir.is_dir()
    assert store.log_dir.is_dir()
    assert store.evaluation_dir.is_dir()
    assert store.manifest_path == store.run_dir / "manifest.json"


def test_manifest_json_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    manifest = ExperimentManifest(
        run_id="run-001",
        experiment_name="demo",
        seed=7,
        code_revision="abc123",
        config={"training": {"steps": 1}},
        dataset={"repo_id": "demo/data", "revision": "v1"},
        policy={"name": "act"},
        environment={"name": "libero"},
        checkpoint={"path": "checkpoints/final.pt"},
        evaluation={"success_rate": 0.5},
    )

    manifest.write_json(path)
    restored = ExperimentManifest.read_json(path)

    assert restored == manifest
    assert json.loads(path.read_text(encoding="utf-8"))["run_id"] == "run-001"


def test_manifest_nested_mappings_are_copied() -> None:
    config = {"steps": 1}
    manifest = ExperimentManifest(run_id="run", experiment_name="demo", seed=0, config=config)

    config["steps"] = 2

    assert manifest.config["steps"] == 1


@pytest.mark.parametrize("run_id", ["../escape", "nested/run", "nested\\run"])
def test_artifact_store_rejects_non_component_run_id(tmp_path: Path, run_id: str) -> None:
    with pytest.raises(ValueError, match="single non-empty path component"):
        ArtifactStore(tmp_path, run_id)
