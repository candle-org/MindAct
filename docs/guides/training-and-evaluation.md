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

A runner should write the manifest before training begins. This preserves the original provenance even if a process stops before producing a checkpoint.

## Dataset, policy, and environment boundaries

Concrete integrations are kept behind small protocols. A runner can accept any object implementing the relevant contract, including test doubles:

- `DatasetAdapter` exposes dataset identity, length, and batch iteration.
- `PolicyAdapter` exposes policy identity, checkpoint loading, and action prediction.
- `EnvironmentAdapter` exposes reset, step, and close operations.
- `Trainer` and `Evaluator` return immutable, provenance-friendly result objects.

This makes it possible to replace a storage backend or simulator without changing experiment bookkeeping.

## Optional integrations

MindAct does not import LeRobot or LIBERO while importing the core package. Request the integration explicitly:

```python
from mindact.integrations.lerobot import load_lerobot
from mindact.integrations.libero import load_libero

lerobot = load_lerobot()
libero = load_libero()
```

If an integration is not installed, MindAct raises an actionable `MissingOptionalDependency` error. Install the corresponding extra described in the [installation guide](../getting-started/installation.md).

## Evaluation records

An evaluator should associate every aggregate result with the training `run_id` and checkpoint used. Store machine-readable metrics under a checkpoint-specific directory, for example:

```text
outputs/<run-id>/evaluation/final/
├── results.json
└── episodes/
```

Use `EvaluationResult.success_rate` for the standard success fraction and retain benchmark-specific metrics in `metrics`. Keep raw episode metadata in `metadata` or separate rollout files instead of embedding large arrays in the manifest.

## Reproducibility checklist

Before comparing two runs, verify:

1. The YAML copy and manifest are present.
2. The random seed is recorded and applied by the runner.
3. Dataset `repo_id`, split, and revision are captured.
4. Policy and environment revisions are captured when available.
5. The checkpoint path or identifier used for evaluation is recorded.
6. Evaluation episode count, task selection, and simulator options are unchanged.
7. Code revision is recorded by the runner when the source is under Git.

The v0.1 CLI only validates configurations. Training and simulator commands will be added once their integration contracts are implemented; users should not interpret the skeleton command as a completed training pipeline.
