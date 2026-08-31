from __future__ import annotations

from pathlib import Path

import pytest

from mindact.configs import ConfigError, ExperimentConfig, TrainingConfig

EXAMPLE_CONFIG = Path(__file__).parents[2] / "configs/experiments/libero-baseline.yaml"


def test_example_config_loads() -> None:
    config = ExperimentConfig.from_yaml(EXAMPLE_CONFIG)

    assert config.name == "libero-baseline"
    assert config.seed == 42
    assert config.dataset.repo_id == "lerobot/aloha_sim_insertion_human"
    assert config.environment.task_suite == "libero_spatial"


def test_config_yaml_round_trip(tmp_path: Path) -> None:
    original = ExperimentConfig.from_yaml(EXAMPLE_CONFIG)
    output = tmp_path / "round-trip.yaml"

    original.to_yaml(output)
    restored = ExperimentConfig.from_yaml(output)

    assert restored.to_dict() == original.to_dict()


def test_unknown_root_field_is_rejected() -> None:
    with pytest.raises(ConfigError, match="unknown field.*root"):
        ExperimentConfig.from_dict(
            {
                "name": "demo",
                "unexpected": True,
                "dataset": {"repo_id": "demo"},
                "policy": {"name": "act"},
                "environment": {"name": "libero"},
                "training": {"steps": 1, "batch_size": 1, "learning_rate": 0.1},
            }
        )


def test_negative_training_steps_are_rejected() -> None:
    with pytest.raises(ConfigError, match="training.steps.*greater than zero"):
        TrainingConfig(steps=-1, batch_size=1, learning_rate=0.1)


def test_missing_required_section_is_rejected() -> None:
    with pytest.raises(ConfigError, match="'environment' must be a mapping"):
        ExperimentConfig.from_dict(
            {
                "name": "demo",
                "dataset": {"repo_id": "demo"},
                "policy": {"name": "act"},
                "training": {"steps": 1, "batch_size": 1, "learning_rate": 0.1},
            }
        )


def test_config_is_immutable() -> None:
    config = ExperimentConfig.from_yaml(EXAMPLE_CONFIG)

    with pytest.raises(AttributeError):
        config.name = "changed"  # type: ignore[misc]
