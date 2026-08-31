from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from mindact import EvaluationRunner, ExperimentConfig


class FakePolicy:
    name = "fake"
    revision = "test"

    def __init__(self) -> None:
        self.loaded: list[str] = []
        self.observations: list[dict[str, Any]] = []

    def predict(self, observation: dict[str, Any]) -> int:
        self.observations.append(observation)
        return 1

    def load(self, checkpoint: str) -> None:
        self.loaded.append(checkpoint)


class FakeEnvironment:
    name = "fake"
    version = "1"

    def __init__(self, *, episodes: list[FakeEnvironment]) -> None:
        self.episodes = episodes
        self.seed: int | None = None
        self.steps = 0
        self.closed = False

    def reset(self, *, seed: int | None = None) -> dict[str, Any]:
        self.seed = seed
        return {"seed": seed}

    def step(self, action: Any) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        self.steps += 1
        return {}, 2.5, True, {"success": True, "action": action}

    def close(self) -> None:
        self.closed = True


def make_config(output_dir: Path, *, episodes: int = 2, max_steps: int | None = None) -> ExperimentConfig:
    return ExperimentConfig.from_dict(
        {
            "name": "fake-eval",
            "seed": 40,
            "output_dir": str(output_dir),
            "dataset": {"repo_id": "fake/data"},
            "policy": {"name": "fake"},
            "environment": {"name": "fake"},
            "training": {"steps": 1, "batch_size": 1, "learning_rate": 0.1},
            "evaluation": {"episodes": episodes, "max_steps": max_steps, "record_video": False},
        }
    )


def test_runner_persists_deterministic_evaluation_artifacts(tmp_path: Path) -> None:
    config = make_config(tmp_path, episodes=2)
    policy = FakePolicy()
    environments: list[FakeEnvironment] = []

    def factory() -> FakeEnvironment:
        environment = FakeEnvironment(episodes=environments)
        environments.append(environment)
        return environment

    result = EvaluationRunner(
        config=config,
        policy=policy,
        environment_factory=factory,
        run_id="run-001",
        checkpoint=tmp_path / "checkpoint.pt",
    ).evaluate()

    assert result.successes == 2
    assert result.success_rate == 1.0
    assert policy.loaded == [str(tmp_path / "checkpoint.pt")]
    assert [environment.seed for environment in environments[1:]] == [40, 41]
    assert all(environment.closed for environment in environments)

    run_dir = tmp_path / "run-001"
    assert (run_dir / "config.yaml").exists()
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["checkpoint"] == {"path": str(tmp_path / "checkpoint.pt")}
    episodes = [json.loads(line) for line in (run_dir / "evaluation/episodes.jsonl").read_text().splitlines()]
    assert [episode["seed"] for episode in episodes] == [40, 41]
    assert json.loads((run_dir / "evaluation/results.json").read_text())["success_rate"] == 1.0


def test_manifest_records_runtime_adapter_identity(tmp_path: Path) -> None:
    """Configured names describe intent; runtime identity shows what actually ran."""
    config = make_config(tmp_path, episodes=1)
    environments: list[FakeEnvironment] = []

    EvaluationRunner(
        config=config,
        policy=FakePolicy(),
        environment_factory=lambda: _track(FakeEnvironment(episodes=environments), environments),
        run_id="run-001",
    ).evaluate()

    manifest = json.loads((tmp_path / "run-001/manifest.json").read_text())
    assert manifest["policy"]["name"] == "fake"
    assert manifest["policy"]["runtime"].endswith("FakePolicy")
    assert "test_evaluation_runner" in manifest["policy"]["runtime"]
    # A lambda factory must still resolve to the concrete environment class.
    assert manifest["environment"]["runtime"].endswith("FakeEnvironment")
    assert manifest["environment"]["version"] == "1"


def _track(environment: FakeEnvironment, seen: list[FakeEnvironment]) -> FakeEnvironment:
    seen.append(environment)
    return environment


def test_runner_rejects_existing_run(tmp_path: Path) -> None:
    config = make_config(tmp_path, episodes=1)
    run_dir = tmp_path / "run-001"
    run_dir.mkdir()
    (run_dir / "sentinel").write_text("keep")

    with pytest.raises(FileExistsError, match="run directory already exists"):
        EvaluationRunner(
            config=config,
            policy=FakePolicy(),
            environment_factory=lambda: FakeEnvironment(episodes=[]),
            run_id="run-001",
        ).evaluate()

    assert (run_dir / "sentinel").read_text() == "keep"


def test_runner_marks_max_step_truncation(tmp_path: Path) -> None:
    class LongEnvironment(FakeEnvironment):
        def step(self, action: Any) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
            self.steps += 1
            return {}, 1.0, False, {}

    environment = LongEnvironment(episodes=[])
    result = EvaluationRunner(
        config=make_config(tmp_path, episodes=1, max_steps=2),
        policy=FakePolicy(),
        environment_factory=lambda: environment,
        run_id="run-001",
    ).evaluate()

    assert result.successes == 0
    episode = json.loads((tmp_path / "run-001/evaluation/episodes.jsonl").read_text())
    assert episode["steps"] == 2
    assert episode["truncated"] is True
    assert environment.closed is True


def test_runner_closes_environment_when_policy_fails(tmp_path: Path) -> None:
    class FailingPolicy(FakePolicy):
        def predict(self, observation: dict[str, Any]) -> int:
            raise RuntimeError("policy failure")

    environment = FakeEnvironment(episodes=[])
    with pytest.raises(RuntimeError, match="policy failure"):
        EvaluationRunner(
            config=make_config(tmp_path, episodes=1),
            policy=FailingPolicy(),
            environment_factory=lambda: environment,
            run_id="run-001",
        ).evaluate()

    assert environment.closed is True
    assert (tmp_path / "run-001/manifest.json").exists()
    assert not (tmp_path / "run-001/evaluation/results.json").exists()


def test_runner_is_structurally_protocol_compatible(tmp_path: Path) -> None:
    from mindact.envs import EnvironmentAdapter
    from mindact.evaluation import Evaluator
    from mindact.policies import PolicyAdapter

    policy = FakePolicy()
    environment = FakeEnvironment(episodes=[])
    runner = EvaluationRunner(
        config=make_config(tmp_path, episodes=1),
        policy=policy,
        environment_factory=lambda: environment,
        run_id="run-001",
    )

    assert isinstance(policy, PolicyAdapter)
    assert isinstance(environment, EnvironmentAdapter)
    assert isinstance(runner, Evaluator)


def test_runner_does_not_import_optional_dependencies(tmp_path: Path) -> None:
    import sys

    before = {name for name in ("torch", "lerobot", "libero") if name in sys.modules}
    EvaluationRunner(
        config=make_config(tmp_path, episodes=1),
        policy=FakePolicy(),
        environment_factory=lambda: FakeEnvironment(episodes=[]),
        run_id="run-001",
    ).evaluate()

    assert {name for name in ("torch", "lerobot", "libero") if name in sys.modules} == before


def test_runner_rejects_non_json_episode_info(tmp_path: Path) -> None:
    class BadEnvironment(FakeEnvironment):
        def step(self, action: Any) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
            return {}, 1.0, True, {"invalid": object()}

    with pytest.raises(ValueError, match="JSON serializable"):
        EvaluationRunner(
            config=make_config(tmp_path, episodes=1),
            policy=FakePolicy(),
            environment_factory=lambda: BadEnvironment(episodes=[]),
            run_id="run-001",
        ).evaluate()


def test_runner_repeated_fixed_run_is_rejected_without_overwrite(tmp_path: Path) -> None:
    config = make_config(tmp_path, episodes=1)
    runner = EvaluationRunner(
        config=config,
        policy=FakePolicy(),
        environment_factory=lambda: FakeEnvironment(episodes=[]),
        run_id="run-001",
    )
    runner.evaluate()
    manifest = (tmp_path / "run-001/manifest.json").read_text()

    with pytest.raises(FileExistsError):
        EvaluationRunner(
            config=config,
            policy=FakePolicy(),
            environment_factory=lambda: FakeEnvironment(episodes=[]),
            run_id="run-001",
        ).evaluate()

    assert (tmp_path / "run-001/manifest.json").read_text() == manifest


def test_manifest_rejects_non_finite_nested_values(tmp_path: Path) -> None:
    from mindact import ExperimentManifest

    with pytest.raises(ValueError, match="finite"):
        ExperimentManifest(
            run_id="run",
            experiment_name="demo",
            seed=1,
            config={"metric": float("nan")},
        ).write_json(tmp_path / "manifest.json")


def test_fake_seed_outcomes_are_reproducible(tmp_path: Path) -> None:
    from mindact.evaluation.fake import FakeEnvironment, FakePolicy

    def run(run_id: str) -> bytes:
        EvaluationRunner(
            config=make_config(tmp_path, episodes=4),
            policy=FakePolicy(),
            environment_factory=FakeEnvironment,
            run_id=run_id,
        ).evaluate()
        return (tmp_path / run_id / "evaluation/episodes.jsonl").read_bytes()

    assert run("first") == run("second")
