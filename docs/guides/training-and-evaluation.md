# Training and evaluation

MindAct treats a training or evaluation run as a reproducible experiment. A YAML file describes the inputs and controls, while an artifact store keeps the resulting configuration, manifest, checkpoints, logs, and evaluation records together.

## Validate a configuration

Start with the example configuration:

```bash
mindact config-check configs/experiments/libero-baseline.yaml
```

A successful check prints the experiment name and exits with status zero. Validation catches missing sections, unknown keys, empty identifiers, and invalid positive-only values before a run starts.

Load a configuration from Python when composing a runner:

```python
from mindact import ExperimentConfig

config = ExperimentConfig.from_yaml("configs/experiments/libero-baseline.yaml")
print(config.dataset.repo_id)
```

Configurations are frozen after construction. To create a modified configuration, build a new instance rather than mutating an existing one. Use `config.to_yaml(path)` to write a portable copy alongside a run.

## Recommended run layout

Create one `ArtifactStore` per run and use its paths consistently:

```python
from mindact import ArtifactStore

store = ArtifactStore(config.output_dir, run_id).create()
print(store.checkpoint_dir)
print(store.evaluation_dir)
```

The resulting layout is intentionally easy to inspect:

```text
outputs/<run-id>/
├── manifest.json
├── config.yaml
├── checkpoints/
├── logs/
└── evaluation/
```

A runner should write the manifest before training begins. This preserves the original provenance even if a process stops before producing a checkpoint. Manifests are write-once: completed evaluation metrics are written to separate artifact files rather than replacing the manifest.

## Dependency-free evaluation smoke test

MindAct includes a small fake policy and environment so the evaluation lifecycle can be tested without installing PyTorch, LeRobot, or LIBERO:

```bash
mindact eval configs/experiments/libero-baseline.yaml \
  --runner fake \
  --run-id smoke-run \
  --output-dir /tmp/mindact-outputs \
  --episodes 2
```

The command creates `config.yaml`, `manifest.json`, `evaluation/results.json`, and `evaluation/episodes.jsonl`. Use a fixed `--run-id` when comparing two runs byte-for-byte; an existing run directory is never overwritten.

The same runner is available as a Python API. Adapters are injected, so a real policy or simulator can be connected without changing experiment bookkeeping:

```python
from mindact import EvaluationRunner

runner = EvaluationRunner(
    config=config,
    policy=policy_adapter,
    environment_factory=lambda: environment_adapter,
    run_id="run-001",
)
result = runner.evaluate()
```

Each episode receives `config.seed + episode_index`. The runner calls `reset(seed=...)`, applies actions until `done` or `evaluation.max_steps`, and always calls `close()` in a `finally` block. A final `info["success"]` value determines the episode success flag.

The fake runner is a smoke path only; it does not represent a trained ACT policy or a LIBERO benchmark result. Real training and simulator adapters remain future work.

## Artifact and provenance contract

A completed run uses this layout:

```text
outputs/<run-id>/
├── config.yaml                 # normalized input configuration
├── manifest.json               # immutable run identity and lineage
├── checkpoints/
├── logs/
└── evaluation/
    ├── results.json            # aggregate metrics
    └── episodes.jsonl          # one JSON object per episode
```

`manifest.json` records the dataset, policy, environment, seed, optional checkpoint, and paths to result artifacts. It is created before the first rollout and refuses to overwrite an existing file. Aggregate and per-episode results are independent files, allowing provenance to remain unchanged after evaluation completes.

Artifact metadata must remain JSON-compatible. Large observations, images, and videos should be stored as separate files and referenced from episode metadata rather than embedded in the manifest.

## Dataset, policy, and environment boundaries

Concrete integrations are kept behind small protocols. A runner can accept any object implementing the relevant contract, including test doubles:

- `DatasetAdapter` exposes dataset identity, length, and batch iteration.
- `PolicyAdapter` exposes policy identity, checkpoint loading, and action prediction.
- `EnvironmentAdapter` exposes reset, step, and close operations.
- `Trainer` and `Evaluator` return immutable, provenance-friendly result objects.

This makes it possible to replace a storage backend or simulator without changing experiment bookkeeping.

## Optional integrations

MindAct does not import LeRobot or LIBERO while importing the core package. Use `mindact doctor` to inspect optional package availability without importing those packages, then request an integration explicitly:

```bash
mindact doctor
```

A `FOUND` result only means that a module is importable; it does not validate MuJoCo rendering, device execution, checkpoint access, or a complete simulator installation.

Request the integration explicitly:

```python
from mindact.integrations.lerobot import load_lerobot
from mindact.integrations.libero import load_libero

lerobot = load_lerobot()
libero = load_libero()
```

If an integration is not installed, MindAct raises an actionable `MissingOptionalDependency` error. Install the corresponding extra described in the [installation guide](../getting-started/installation.md).

## Evaluation records

Every aggregate result is associated with the run `run_id` and the checkpoint used, both recorded in `manifest.json`. Aggregate metrics live in `evaluation/results.json` and per-episode records in `evaluation/episodes.jsonl`, one JSON object per line:

```json
{"episode_id": "episode-0000", "seed": 40, "success": true, "reward": 1.0, "steps": 1, "truncated": false, "metadata": {}}
```

Use `EvaluationResult.success_rate` for the standard success fraction and retain benchmark-specific metrics in `metrics`. Keep raw episode metadata in `EpisodeRecord.metadata` or separate rollout files instead of embedding large arrays in the manifest.

## Reproducibility checklist

Before comparing two runs, verify:

1. The YAML copy and manifest are present.
2. The random seed is recorded and applied by the runner.
3. Dataset `repo_id`, split, and revision are captured.
4. Policy and environment revisions are captured when available.
5. The checkpoint path or identifier used for evaluation is recorded.
6. Evaluation episode count, task selection, and simulator options are unchanged.
7. Code revision is recorded by the runner when the source is under Git.

The v0.1 CLI covers configuration validation and the fake evaluation path. A `train` command and real simulator adapters will be added once their integration contracts are implemented; the fake runner should not be interpreted as a completed training or benchmark pipeline.
