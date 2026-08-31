# MindAct Project - Claude Code Configuration

## Project Overview

MindAct is a PyTorch and Hugging Face native training and evaluation toolkit for embodied policies. It provides reproducible experiment tracking for imitation learning on desktop manipulation tasks, integrating LeRobot datasets/policies with LIBERO simulation benchmarks.

## Directory Structure

```
mindnlp/  (repository root)
├── .claude/
│   ├── settings.json
│   └── projects/
├── src/
│   └── mindact/
│       ├── cli/              # Command-line interface
│       ├── configs/          # Configuration schemas
│       ├── datasets/         # Dataset adapter protocols
│       ├── envs/             # Environment adapter protocols
│       ├── evaluation/       # Evaluation protocols
│       ├── experiments/      # Manifest and artifact management
│       ├── integrations/     # Lazy loading wrappers for lerobot/libero
│       ├── policies/         # Policy adapter protocols
│       ├── training/         # Training protocols
│       └── utils/            # Optional import handling
├── configs/
│   ├── experiments/          # YAML experiment configurations
│   └── README.md
├── examples/
│   └── libero/               # LIBERO integration examples
├── tests/
│   ├── unit/                 # Unit tests for core modules
│   └── integration/          # Integration tests with optional deps
├── docs/
│   ├── getting-started/
│   ├── concepts/
│   ├── guides/
│   └── contributing.md
├── benchmarks/               # Performance benchmarks
├── pyproject.toml            # PEP 621 package metadata
├── README.md
├── CLAUDE.md                 # This file
├── LICENSE
└── NOTICE
```

## Current Status (as of 2026-08-31)

**Version**: 0.1.0 (skeleton release)  
**Branch**: feature/mindact-v0.1 (active development)  
**Legacy branch**: legacy (preserves old MindNLP/MindTorch code)

### Implemented

- Protocol-based adapter interfaces (datasets, policies, environments, trainers, evaluators)
- Typed configuration system with YAML serialization
- Experiment provenance tracking (manifests + artifacts)
- Optional dependency handling (torch, lerobot, libero as extras)
- CLI skeleton with config validation
- Example YAML configurations
- Unit tests and skip-clean optional integration smoke tests
- MkDocs documentation skeleton and contributor guide
- Python 3.12/3.13 CI for tests, linting, and package builds

### Not Yet Implemented

- Training loop (Trainer protocol implementation)
- Evaluation loop (Evaluator protocol implementation)
- LeRobot dataset adapter
- LeRobot policy adapter
- LIBERO environment adapter
- Checkpoint loading/saving
- Metrics logging

---

## Core Design Principles

### 1. Protocol-Based Architecture

**CRITICAL**: All integration boundaries use `@runtime_checkable Protocol` from `typing`, not ABC or concrete base classes.

- **Why**: Allows external libraries to satisfy MindAct contracts without subclassing
- **Example**: LeRobot policies can be wrapped as `PolicyAdapter` without modification
- **Rule**: Never require inheritance from MindAct base classes

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class PolicyAdapter(Protocol):
    """Contract for policy implementations."""
    name: str
    revision: str | None
    
    def predict(self, observation: dict[str, Any]) -> dict[str, Any]: ...
    def load(self, checkpoint: str | Path) -> None: ...
```

### 2. Optional Dependencies

**CRITICAL**: MindAct core has ZERO required dependencies except numpy and pyyaml. All ML frameworks are optional extras.

- **Never** import `torch`, `lerobot`, or `libero` at module level
- **Always** use `require_module()` from `mindact.utils.imports` at runtime
- **Rationale**: Users can install only what they need; config validation works without PyTorch

```python
# BAD: Eager import
import torch
from lerobot import Dataset

# GOOD: Lazy loading
def load_lerobot_dataset(config: DatasetConfig):
    lerobot = require_module("lerobot", feature="LeRobot dataset loading")
    return lerobot.Dataset(config.source, split=config.split)
```

### 3. Frozen Dataclasses for All Configurations

**CRITICAL**: All configuration and result objects are frozen dataclasses. No mutable state.

- **Why**: Prevents accidental modification after validation; enables safe caching
- **Rule**: Use `@dataclass(frozen=True, kw_only=True)` for all configs and results
- **Exception**: None

```python
from dataclasses import dataclass

@dataclass(frozen=True, kw_only=True)
class TrainingConfig:
    """Training hyperparameters."""
    steps: int
    batch_size: int
    learning_rate: float
    checkpoint_every: int
```

### 4. Experiment Provenance

**CRITICAL**: Every training run generates a manifest tracking exact dataset/policy/checkpoint/evaluation lineage.

- **Manifest location**: `experiments/<run-id>/manifest.json`
- **Required fields**: experiment name, seed, dataset (source, revision, split), policy (architecture, checkpoint), training config, evaluation config, run_id, timestamp
- **Rule**: Manifests are write-once, never modified after creation
- **Format**: JSON for machine readability, YAML for human-editable configs

---

## Development Workflow

### Branch Strategy

- **master**: Stable releases (currently empty, waiting for v0.1 commit)
- **legacy**: Old MindNLP/MindTorch code (read-only archive)
- **feature/mindact-v0.1**: Current development branch
- **Feature branches**: `feature/<name>` for new features

### Git Remotes

- **origin**: Your fork/development repository (push target)
- **ms**: Upstream repository (pull source, if applicable)

### Pull Request Workflow

When creating PRs:

1. Rebase onto target branch (master or feature/mindact-v0.1)
2. Squash all commits into ONE commit with a descriptive message
3. Push with `--force-with-lease` after rebase
4. Create PR with clear description of changes

### Testing

```bash
# Activate environment
source ~/miniconda3/bin/activate mindnlp  # or your preferred env

# Run unit tests (no optional deps required)
pytest tests/unit/ -v

# Run integration tests (requires torch, lerobot, libero)
pytest tests/integration/ -v

# Run all tests
pytest -v
```

### Code Quality

- **Type hints**: All public functions must have full type annotations
- **Docstrings**: NumPy style for all public APIs
- **Linting**: `ruff check src/ tests/`
- **Formatting**: `ruff format src/ tests/`

---

## Important Constraints

### For All Code Changes

- **Never** add eager imports of optional dependencies
- **Never** use mutable configuration objects
- **Never** skip manifest generation in training runs
- **Always** validate YAML configs against typed schemas before use
- **Always** preserve provenance (dataset revision, policy checkpoint, config)

### For Integration Code

- **LeRobot integration**: Lives in `src/mindact/integrations/lerobot.py`
- **LIBERO integration**: Lives in `src/mindact/integrations/libero.py`
- **Rule**: Integration modules contain ONLY lazy loading wrappers, not adapter implementations
- **Adapter implementations**: Live in `src/mindact/datasets/`, `src/mindact/policies/`, `src/mindact/envs/`

### For Configuration Changes

- **Never** add fields to config schemas without default values (breaks existing YAMLs)
- **Always** use frozen dataclasses, not plain dicts or Pydantic models
- **Validation**: Happens in `__post_init__` methods, not external validators

---

## CLI Commands

### Current

```bash
# Validate experiment configuration
mindact config-check <yaml-path>
```

### Planned

```bash
# Train a policy
mindact train <yaml-path> [--output-dir DIR]

# Evaluate a checkpoint
mindact eval <checkpoint-path> --config <yaml-path> [--episodes N]

# List available datasets
mindact datasets list [--source lerobot|huggingface]

# List available policies
mindact policies list

# Show experiment manifest
mindact manifest show <run-id>
```

---

## Troubleshooting

### Import Errors

- **ModuleNotFoundError for torch/lerobot/libero**: Install missing extras with `pip install -e ".[torch,lerobot,libero]"`
- **Import fails without helpful message**: Check `require_module()` call includes `feature=` parameter

### Configuration Errors

- **ConfigError during YAML load**: Check YAML syntax, required fields, and value types
- **Dataclass FrozenInstanceError**: Don't try to modify config after creation; create new config instead

### Git Issues

- **Detached HEAD**: Check out a branch with `git checkout feature/mindact-v0.1`
- **Merge conflicts**: MindAct is fresh; conflicts should only occur during rebases onto master

---

## External Resources

- **LeRobot**: https://github.com/huggingface/lerobot
- **LIBERO**: https://github.com/Lifelong-Robot-Learning/LIBERO
- **PyTorch**: https://pytorch.org/docs/stable/index.html
- **Hugging Face Hub**: https://huggingface.co/docs/hub/index

---

## Repository History

This repository was originally [MindNLP](https://github.com/mindspore-lab/mindnlp), a MindSpore-based NLP library for Ascend/GPU/CPU. The legacy codebase (including MindTorch v1/v2) is preserved in the `legacy` branch at commit b288bacf.

MindAct represents a complete product pivot to embodied AI, using PyTorch and Hugging Face as the native stack. The repository was retained to preserve stars (920) and community presence while changing direction entirely.

**Important**: Old MindNLP/MindTorch rules, constraints, and workflows do NOT apply to MindAct development.
