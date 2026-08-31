# MindAct

**A PyTorch and Hugging Face native training and evaluation toolkit for embodied policies**

MindAct is a reproducible training and evaluation framework for imitation learning policies on desktop manipulation tasks. It integrates [LeRobot](https://github.com/huggingface/lerobot) datasets and policies with [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) simulation benchmarks, providing experiment provenance tracking and standardized evaluation protocols.

## Features

- **Protocol-based architecture**: Minimal adapter interfaces for datasets, policies, and environments
- **Reproducible experiments**: YAML configurations with manifest-tracked provenance
- **Optional dependencies**: Lazy loading of torch, lerobot, and libero via extras
- **Artifact management**: Conventional directory structure for checkpoints, logs, and results
- **Type-safe**: Frozen dataclasses and runtime protocols throughout

## Quick Start

### Installation

```bash
# Basic installation (configuration and CLI only)
pip install -e .

# With PyTorch
pip install -e ".[torch]"

# With LeRobot datasets and policies
pip install -e ".[lerobot]"

# With LIBERO simulation environments
pip install -e ".[libero]"

# All integrations
pip install -e ".[torch,lerobot,libero]"

# Development tools
pip install -e ".[dev]"
```

### Run an Example

```bash
# Validate configuration
mindact config-check configs/experiments/libero-baseline.yaml
```

`config-check` is the only command in v0.1. Training and evaluation commands arrive with their adapter implementations.

### Configuration Format

```yaml
name: libero-baseline
seed: 42
output_dir: outputs

dataset:
  repo_id: lerobot/aloha_sim_insertion_human
  revision: null
  split: train

policy:
  name: act
  pretrained_model: null

environment:
  name: libero
  task_suite: libero_spatial
  task_ids: []

training:
  steps: 10000
  batch_size: 8
  learning_rate: 0.0001
  log_every: 100
  checkpoint_every: 2000

evaluation:
  episodes: 20
  max_steps: 500
  record_video: false
```

## Project Status

MindAct v0.1 is the initial skeleton release. Core interfaces and configuration system are stable. Implementation priorities:

**Phase B (current)**: Unified reproducible training and evaluation  
**Phase A (future)**: Trajectory quality diagnostics

See `docs/` for architecture details and contribution guidelines.

## Repository History

This repository was originally [MindNLP](https://github.com/mindspore-lab/mindnlp), a MindSpore-based NLP library. The legacy codebase is preserved in the `legacy` branch. MindAct represents a complete pivot to embodied AI with PyTorch and Hugging Face as the native stack.

## License

Apache License 2.0. See LICENSE and NOTICE for details.
