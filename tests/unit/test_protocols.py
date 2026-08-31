from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from mindact.datasets import DatasetAdapter, DatasetInfo
from mindact.envs import EnvironmentAdapter
from mindact.evaluation import EvaluationResult, Evaluator
from mindact.policies import PolicyAdapter
from mindact.training import Trainer, TrainResult


class FakeDataset:
    info = DatasetInfo(repo_id="demo/data")

    def __len__(self) -> int:
        return 1

    def iter_batches(self, batch_size: int) -> list[Mapping[str, Any]]:
        return [{"size": batch_size}]


class FakePolicy:
    name = "fake"
    revision = None

    def predict(self, observation: Mapping[str, Any]) -> int:
        return 0

    def load(self, checkpoint: str) -> None:
        pass


class FakeEnvironment:
    name = "fake-env"
    version = "1"

    def reset(self, *, seed: int | None = None) -> Mapping[str, Any]:
        return {"seed": seed}

    def step(self, action: Any) -> tuple[Mapping[str, Any], float, bool, Mapping[str, Any]]:
        return {}, 0.0, True, {}

    def close(self) -> None:
        pass


class FakeTrainer:
    def train(self) -> TrainResult:
        return TrainResult(run_id="run", steps=1)


class FakeEvaluator:
    def evaluate(self) -> EvaluationResult:
        return EvaluationResult(run_id="run", episodes=1, successes=1)


def test_protocols_accept_structural_implementations() -> None:
    assert isinstance(FakeDataset(), DatasetAdapter)
    assert isinstance(FakePolicy(), PolicyAdapter)
    assert isinstance(FakeEnvironment(), EnvironmentAdapter)
    assert isinstance(FakeTrainer(), Trainer)
    assert isinstance(FakeEvaluator(), Evaluator)
