"""Framework-neutral deterministic evaluation runner."""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import yaml

from mindact.configs import ExperimentConfig
from mindact.envs import EnvironmentAdapter
from mindact.evaluation.base import EpisodeRecord, EvaluationResult
from mindact.experiments import ArtifactStore, ExperimentManifest
from mindact.policies import PolicyAdapter

__all__ = ["EnvironmentFactory", "EvaluationRunner"]

EnvironmentFactory = Callable[[], EnvironmentAdapter]


def _json_safe(value: Any, field_name: str) -> Any:
    """Validate a value and return it unchanged when JSON serializable."""
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be JSON serializable: {exc}") from exc
    return value


class EvaluationRunner:
    """Run deterministic rollouts against injected policy and environment adapters."""

    def __init__(
        self,
        *,
        config: ExperimentConfig,
        policy: PolicyAdapter,
        environment_factory: EnvironmentFactory,
        store: ArtifactStore | None = None,
        run_id: str | None = None,
        checkpoint: str | Path | None = None,
        code_revision: str | None = None,
        environment_version: str | None = None,
    ) -> None:
        self.config = config
        self.policy = policy
        self.environment_factory = environment_factory
        self.checkpoint = None if checkpoint is None else str(checkpoint)
        self.code_revision = code_revision
        self.environment_version = environment_version
        self.run_id = run_id or f"eval-{uuid.uuid4().hex[:12]}"
        if not callable(environment_factory):
            raise TypeError("environment_factory must be callable")
        if not callable(getattr(policy, "predict", None)) or not callable(getattr(policy, "load", None)):
            raise TypeError("policy must implement callable predict() and load()")
        if self.checkpoint is not None and not self.checkpoint.strip():
            raise ValueError("checkpoint must be a non-empty path or reference")
        if code_revision is not None and not code_revision.strip():
            raise ValueError("code_revision must be a non-empty string or None")
        if environment_version is not None and not environment_version.strip():
            raise ValueError("environment_version must be a non-empty string or None")
        self.store = store or ArtifactStore(root=config.output_dir, run_id=self.run_id)
        if self.store.run_id != self.run_id:
            raise ValueError("store.run_id must match run_id")

    def evaluate(self) -> EvaluationResult:
        """Execute all configured episodes and persist their provenance artifacts."""
        self._prepare_run()
        if self.checkpoint is not None:
            self.policy.load(self.checkpoint)

        records: list[EpisodeRecord] = []
        for episode_index in range(self.config.evaluation.episodes):
            records.append(self._evaluate_episode(episode_index))

        result = self._aggregate(records)
        self.store.write_jsonl(self.store.episodes_path, [record.to_dict() for record in records])
        self.store.write_json(self.store.results_path, result.to_dict())
        return result

    def _inspect_environment(self) -> tuple[str | None, str]:
        """Read runtime identity from one environment without consuming an episode."""
        environment = self.environment_factory()
        try:
            return environment.version, self._runtime_identity(environment)
        finally:
            environment.close()

    @staticmethod
    def _runtime_identity(adapter: object) -> str:
        """Return the import path of the adapter that actually produced a run.

        Configuration names the intended dataset, policy and simulator, which is
        not necessarily what executed. Recording the concrete implementation
        keeps a fake or stub run from being mistaken for a real benchmark.
        """
        target = adapter if hasattr(adapter, "__qualname__") else type(adapter)
        module = getattr(target, "__module__", None) or "?"
        return f"{module}.{target.__qualname__}"

    def _prepare_run(self) -> None:
        if self.store.run_dir.exists():
            raise FileExistsError(f"run directory already exists: '{self.store.run_dir}'")
        self.store.create()
        self.store.write_text(
            self.store.config_path,
            yaml.safe_dump(self.config.to_dict(), sort_keys=False),
        )
        checkpoint = None
        if self.checkpoint is not None:
            checkpoint = {"path": self.checkpoint}
        inspected_version, environment_runtime = self._inspect_environment()
        manifest = ExperimentManifest(
            run_id=self.run_id,
            experiment_name=self.config.name,
            seed=self.config.seed,
            config=self.config.to_dict(),
            code_revision=self.code_revision,
            dataset={
                "repo_id": self.config.dataset.repo_id,
                "revision": self.config.dataset.revision,
                "split": self.config.dataset.split,
            },
            policy={
                "name": self.config.policy.name,
                "revision": self.config.policy.revision,
                "pretrained_model": self.config.policy.pretrained_model,
                "runtime": self._runtime_identity(self.policy),
            },
            environment={
                "name": self.config.environment.name,
                "version": self.environment_version or inspected_version,
                "task_suite": self.config.environment.task_suite,
                "task_ids": list(self.config.environment.task_ids),
                "runtime": environment_runtime,
            },
            checkpoint=checkpoint,
            evaluation={
                "results_path": str(self.store.results_path.relative_to(self.store.run_dir)),
                "episodes_path": str(self.store.episodes_path.relative_to(self.store.run_dir)),
            },
        )
        manifest.write_json(self.store.manifest_path)

    def _evaluate_episode(self, episode_index: int) -> EpisodeRecord:
        episode_seed = self.config.seed + episode_index
        environment = self.environment_factory()
        try:
            observation = environment.reset(seed=episode_seed)
            total_reward = 0.0
            steps = 0
            done = False
            final_info: Mapping[str, Any] = {}
            max_steps = self.config.evaluation.max_steps
            while not done and (max_steps is None or steps < max_steps):
                action = self.policy.predict(observation)
                observation, reward, done, info = environment.step(action)
                if not isinstance(info, Mapping):
                    raise TypeError("environment.step() must return a mapping as its info value")
                if isinstance(done, bool) is False:
                    raise TypeError("environment.step() must return a boolean done value")
                total_reward += self._reward_value(reward)
                steps += 1
                final_info = info
            truncated = not done
            success = bool(final_info.get("success", False))
            metadata = {"info": _json_safe(dict(final_info), "episode metadata")}
            return EpisodeRecord(
                episode_id=f"episode-{episode_index:04d}",
                seed=episode_seed,
                success=success,
                reward=total_reward,
                steps=steps,
                truncated=truncated,
                metadata=metadata,
            )
        finally:
            environment.close()

    @staticmethod
    def _reward_value(reward: Any) -> float:
        if isinstance(reward, bool) or not isinstance(reward, (int, float)):
            raise TypeError("environment.step() must return a numeric reward")
        value = float(reward)
        if value != value or value in {float("inf"), float("-inf")}:
            raise ValueError("environment.step() must return a finite reward")
        return value

    def _aggregate(self, records: list[EpisodeRecord]) -> EvaluationResult:
        total_reward = sum(record.reward for record in records)
        total_steps = sum(record.steps for record in records)
        return EvaluationResult(
            run_id=self.run_id,
            episodes=len(records),
            successes=sum(record.success for record in records),
            metrics={
                "mean_reward": total_reward / len(records),
                "mean_steps": total_steps / len(records),
            },
            metadata={
                "results_path": str(self.store.results_path),
                "episodes_path": str(self.store.episodes_path),
                "checkpoint": self.checkpoint,
            },
        )
