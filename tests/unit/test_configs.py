from __future__ import annotations

from pathlib import Path

import pytest

from mindact.configs import (
    ConfigError,
    DatasetConfig,
    EvaluationConfig,
    ExperimentConfig,
    TrainingConfig,
)

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


def test_boolean_is_not_accepted_as_positive_int() -> None:
    with pytest.raises(ConfigError, match="training.batch_size.*greater than zero"):
        TrainingConfig(steps=1, batch_size=True, learning_rate=0.1)  # type: ignore[arg-type]


def test_non_finite_learning_rate_is_rejected() -> None:
    with pytest.raises(ConfigError, match="training.learning_rate.*greater than zero"):
        TrainingConfig(steps=1, batch_size=1, learning_rate=float("inf"))


def test_empty_optional_string_is_rejected() -> None:
    with pytest.raises(ConfigError, match="dataset.revision.*non-empty string"):
        DatasetConfig(repo_id="demo", revision="  ")


def test_non_boolean_record_video_is_rejected() -> None:
    with pytest.raises(ConfigError, match="evaluation.record_video.*boolean"):
        EvaluationConfig(record_video=1)  # type: ignore[arg-type]


def test_nested_options_are_frozen_and_snapshotted() -> None:
    source = {"nested": {"values": [1, 2]}}
    config = DatasetConfig(repo_id="demo", options=source)

    source["nested"]["values"].append(3)

    assert config.options["nested"]["values"] == (1, 2)
    with pytest.raises(TypeError):
        config.options["nested"] = {}  # type: ignore[index]


def test_non_serializable_option_value_is_rejected() -> None:
    with pytest.raises(ConfigError, match="not YAML/JSON compatible"):
        DatasetConfig(repo_id="demo", options={"handle": object()})


def test_empty_output_dir_is_rejected() -> None:
    with pytest.raises(ConfigError, match="output_dir.*must not be empty"):
        ExperimentConfig.from_dict(
            {
                "name": "demo",
                "output_dir": "",
                "dataset": {"repo_id": "demo"},
                "policy": {"name": "act"},
                "environment": {"name": "libero"},
                "training": {"steps": 1, "batch_size": 1, "learning_rate": 0.1},
            }
        )
