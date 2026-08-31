from __future__ import annotations

import json
from dataclasses import FrozenInstanceError

import pytest

from mindact.envs import EpisodeResult
from mindact.evaluation import EpisodeRecord, EvaluationResult
from mindact.training import TrainResult

RESULT_FACTORIES = [
    lambda metadata: TrainResult(run_id="run", steps=1, metadata=metadata),
    lambda metadata: EpisodeResult(
        episode_id="episode-0", success=True, reward=1.0, steps=1, metadata=metadata
    ),
    lambda metadata: EpisodeRecord(
        episode_id="episode-0", seed=0, success=True, reward=1.0, steps=1, metadata=metadata
    ),
    lambda metadata: EvaluationResult(run_id="run", episodes=1, successes=1, metadata=metadata),
]


@pytest.mark.parametrize("factory", RESULT_FACTORIES)
def test_results_are_frozen(factory) -> None:
    result = factory({})

    with pytest.raises(FrozenInstanceError):
        result.metadata = {}  # type: ignore[misc]


@pytest.mark.parametrize("factory", RESULT_FACTORIES)
def test_result_metadata_is_deeply_frozen(factory) -> None:
    source = {"nested": {"values": [1, 2]}}
    result = factory(source)

    source["nested"]["values"].append(3)

    assert result.metadata["nested"]["values"] == (1, 2)
    with pytest.raises(TypeError):
        result.metadata["nested"] = {}  # type: ignore[index]


@pytest.mark.parametrize("factory", RESULT_FACTORIES)
def test_result_to_dict_is_json_serializable(factory) -> None:
    result = factory({"nested": {"values": [1, 2]}})

    payload = json.loads(json.dumps(result.to_dict()))

    assert payload["metadata"]["nested"]["values"] == [1, 2]


@pytest.mark.parametrize("factory", RESULT_FACTORIES)
def test_result_rejects_non_finite_metadata(factory) -> None:
    with pytest.raises(ValueError, match="finite JSON-compatible"):
        factory({"loss": float("nan")})


@pytest.mark.parametrize("factory", RESULT_FACTORIES)
def test_result_rejects_non_string_metadata_keys(factory) -> None:
    with pytest.raises(ValueError, match="keys must be strings"):
        factory({1: "one"})


def test_results_reject_positional_arguments() -> None:
    with pytest.raises(TypeError):
        TrainResult("run", 1)  # type: ignore[misc]
    with pytest.raises(TypeError):
        EpisodeResult("episode-0", True, 1.0, 1)  # type: ignore[misc]
    with pytest.raises(TypeError):
        EvaluationResult("run", 1, 1)  # type: ignore[misc]


def test_train_result_rejects_negative_steps() -> None:
    with pytest.raises(ValueError, match="steps must be a non-negative integer"):
        TrainResult(run_id="run", steps=-1)


def test_train_result_rejects_non_finite_metric() -> None:
    with pytest.raises(ValueError, match="finite JSON-compatible"):
        TrainResult(run_id="run", steps=1, metrics={"loss": float("inf")})


def test_episode_result_rejects_blank_identifier() -> None:
    with pytest.raises(ValueError, match="episode_id must be a non-empty string"):
        EpisodeResult(episode_id="  ", success=True, reward=0.0, steps=0)


def test_evaluation_result_rejects_more_successes_than_episodes() -> None:
    with pytest.raises(ValueError, match="successes must be between zero and episodes"):
        EvaluationResult(run_id="run", episodes=1, successes=2)


def test_evaluation_result_to_dict_keeps_explicit_success_rate() -> None:
    result = EvaluationResult(run_id="run", episodes=4, successes=1)

    assert result.to_dict()["metrics"]["success_rate"] == 0.25
