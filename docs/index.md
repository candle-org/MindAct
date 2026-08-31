# MindAct

MindAct is a PyTorch- and Hugging Face-native toolkit for reproducible training and evaluation of embodied policies on desktop manipulation tasks.

## What MindAct provides

- Typed, validated YAML experiment configurations
- Protocol-based boundaries for datasets, policies, environments, trainers, and evaluators
- Optional LeRobot and LIBERO integrations without import-time heavyweight dependencies
- Portable experiment manifests that preserve dataset, policy, checkpoint, and evaluation lineage
- A conventional artifact layout for checkpoints, logs, rollouts, and metrics

## Start here

1. [Install MindAct](getting-started/installation.md)
2. Run `mindact config-check configs/experiments/libero-baseline.yaml`
3. Read the [architecture overview](concepts/architecture.md)
4. Follow the [training and evaluation guide](guides/training-and-evaluation.md)

MindAct v0.1 is a foundation release. The public interfaces establish reproducible experiment boundaries; concrete training and simulator adapters are being developed incrementally.
