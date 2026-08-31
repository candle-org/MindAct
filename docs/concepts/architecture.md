# Architecture

MindAct is designed around **reproducible experiments** and **loose coupling**. This document explains the core design decisions.

## Design Principles

### 1. Protocol-Based Interfaces

MindAct uses `typing.Protocol` for all integration boundaries instead of abstract base classes:

```python
from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

@runtime_checkable
class PolicyAdapter(Protocol):
    """Inference and checkpoint boundary for an embodied policy."""

    @property
    def name(self) -> str: ...

    @property
    def revision(self) -> str | None: ...

    def predict(self, observation: Mapping[str, Any]) -> Any: ...
    def load(self, checkpoint: str) -> None: ...
```

**Why protocols?**

- External libraries can satisfy MindAct contracts without subclassing
- No inheritance required; duck typing with type checking
- LeRobot policies can be wrapped without modifying LeRobot source
- Enables testing with simple mock objects

**Alternative considered:** Abstract base classes (ABC)  
**Why rejected:** Forces inheritance; couples MindAct to implementation classes

### 2. Frozen Dataclasses

All configuration and result objects are frozen dataclasses:

```python
from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Immutable experiment configuration."""
    name: str
    dataset: DatasetConfig
    policy: PolicyConfig
    environment: EnvironmentConfig
    training: TrainingConfig
    # ...
```

**Why frozen?**

- Prevents accidental mutation after validation
- Enables safe caching and sharing across threads
- Makes it clear when a new configuration is created vs. modified
- Reduces entire class of bugs (mutable default arguments, unexpected state changes)

**Alternative considered:** Pydantic models  
**Why rejected:** Heavy dependency for core data structures; frozen dataclasses are stdlib

### 3. Optional Dependencies

MindAct core has only two required dependencies: `numpy` and `pyyaml`. All ML frameworks are optional extras:

```bash
pip install .                   # Core only
pip install ".[torch]"          # + PyTorch
pip install ".[lerobot]"        # + LeRobot (implies torch)
pip install ".[libero]"         # + LIBERO (implies torch)
```

**Implementation:**

```python
# mindact/utils/imports.py
def require_module(module: str, *, feature: str | None = None) -> ModuleType:
    """Import ``module`` or raise an actionable error."""
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        what = feature or f"the {module} integration"
        raise MissingOptionalDependency(
            f"{what} requires the '{module}' package, which is not installed. "
            f"Install it with: {_install_hint(module)}"
        ) from exc
```

**Why lazy loading?**

- Users install only what they need
- Configuration validation works without PyTorch (lightweight CI)
- Reduces installation time and disk usage for researchers focused on analysis
- Prevents version conflicts between torch/tf/jax users

**Alternative considered:** Make torch a required dependency  
**Why rejected:** Forces researchers to install 2GB+ package for config validation

### 4. Experiment Provenance

Every training run generates a manifest tracking exact provenance:

```json
{
  "run_id": "20260831T143022-abc123",
  "experiment_name": "libero-baseline",
  "seed": 42,
  "code_revision": "abc123",
  "config": {"training": {"steps": 10000}},
  "dataset": {
    "repo_id": "lerobot/aloha_sim_insertion_human",
    "revision": "abc123",
    "split": "train"
  },
  "policy": {"name": "act", "revision": null},
  "environment": {"name": "libero", "task_suite": "libero_spatial"},
  "checkpoint": {"path": "checkpoints/final.pt"},
  "evaluation": {"success_rate": 0.42}
}
```

**Why manifests?**

- Reproducibility: Exact dataset version + policy checkpoint + hyperparameters
- Traceability: Map evaluation results back to training conditions
- Auditability: Verify no silent config changes between runs
- Debugging: Compare manifests to find why two runs differ

**Artifact directory structure:**

```
outputs/
└── 20260831T143022-abc123/
    ├── manifest.json         # Immutable provenance record
    ├── config.yaml           # Human-readable config copy
    ├── checkpoints/
    │   ├── step-5000.pt
    │   ├── step-10000.pt
    │   └── final.pt
    ├── logs/
    │   ├── train.log
    │   └── tensorboard/
    └── evaluation/
        ├── step-5000/
        │   ├── results.json
        │   └── episodes/
        └── final/
            ├── results.json
            └── episodes/
```

**Design decision:** Manifests are write-once JSON, configs are editable YAML  
**Rationale:** Machines read manifests (provenance), humans edit configs (experiments)

## Component Architecture

### Configuration System

**Files:**
- `src/mindact/configs/schema.py` - Typed dataclass schemas
- `src/mindact/configs/__init__.py` - Public API

**Key types:**
- `ExperimentConfig` - Top-level experiment definition
- `DatasetConfig` - Dataset source, split, revision
- `PolicyConfig` - Policy architecture, checkpoint
- `EnvironmentConfig` - Simulation task
- `TrainingConfig` - Hyperparameters
- `EvaluationConfig` - Eval protocol

**Validation:** `__post_init__` methods check:
- Non-empty strings
- Positive integers/floats
- Valid enum values
- Cross-field constraints

**YAML serialization:**
```python
# Load
config = ExperimentConfig.from_yaml("config.yaml")

# Save
config.to_yaml("output.yaml")
```

### Adapter Protocols

**Files:**
- `src/mindact/datasets/base.py` - `DatasetAdapter` protocol
- `src/mindact/policies/base.py` - `PolicyAdapter` protocol
- `src/mindact/envs/base.py` - `EnvironmentAdapter` protocol
- `src/mindact/training/base.py` - `Trainer` protocol
- `src/mindact/evaluation/base.py` - `Evaluator` protocol

**Contract example:**

```python
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

@dataclass(frozen=True, slots=True)
class DatasetInfo:
    """Stable identity and metadata captured in an experiment manifest."""
    repo_id: str
    revision: str | None = None
    split: str = "train"
    features: Mapping[str, str] = field(default_factory=dict)

@runtime_checkable
class DatasetAdapter(Protocol):
    """Contract for dataset implementations."""
    
    @property
    def info(self) -> DatasetInfo: ...
    
    def __len__(self) -> int: ...
    
    def iter_batches(self, batch_size: int) -> Iterable[Mapping[str, Any]]: ...
```

**Implementation approach:**

1. Check if upstream library object satisfies protocol directly (duck typing)
2. If not, write a thin adapter wrapper
3. Adapters live in `src/mindact/datasets/`, `src/mindact/policies/`, or `src/mindact/envs/`
4. Integration modules (`src/mindact/integrations/`) contain only lazy loading logic

### Experiment Management

**Files:**
- `src/mindact/experiments/manifest.py` - Provenance tracking
- `src/mindact/experiments/artifacts.py` - Directory structure

**Key types:**
- `ExperimentManifest` - Immutable provenance record
- `ArtifactStore` - Path helpers for experiment outputs

**Lifecycle:**

1. User creates `ExperimentConfig` from YAML
2. Trainer generates `run_id` and creates `ArtifactStore`
3. Trainer writes `manifest.json` before training starts
4. Checkpoints saved to `store.checkpoint_dir`
5. Logs written to `store.log_dir`
6. Evaluator reads manifest, saves results to `store.evaluation_dir`

**Manifest immutability:**
- Written once at training start
- Never modified (even if training crashes)
- If config changes mid-run, that's a bug to fix, not a manifest to update

### CLI

**Files:**
- `src/mindact/cli/main.py` - Entry point
- `src/mindact/cli/` (planned) - Subcommand implementations

**Current commands:**
- `mindact config-check` - Validate YAML configuration

**Planned commands:**
- `mindact train` - Run training loop
- `mindact eval` - Evaluate checkpoint
- `mindact datasets list` - List available datasets
- `mindact policies list` - List available policies
- `mindact manifest show` - Display experiment manifest

## Error Handling

### Configuration Errors

Raised as `ConfigError` (subclass of `ValueError`):

```python
from mindact.configs import ConfigError

try:
    config = ExperimentConfig.from_yaml("bad.yaml")
except ConfigError as e:
    print(f"Configuration error: {e}")
```

**User-facing error messages:**
- Clear: "training.steps must be positive, got -100"
- Actionable: "dataset.source cannot be empty"
- No stack traces for validation errors (not a bug, user mistake)

### Import Errors

Raised when optional dependency missing:

```python
from mindact.utils.imports import require_module, MissingOptionalDependency

try:
    torch = require_module("torch", feature="training")
except MissingOptionalDependency as e:
    print(e)
    # Output: "training requires the 'torch' package, which is not installed. Install it with: pip install "mindact[torch]""
```

**User-facing error messages:**
- Tell user what's missing: "torch is required"
- Tell user how to fix it: "Install with: pip install mindact[torch]"
- No generic "ModuleNotFoundError" stack traces

## Testing Strategy

### Unit Tests (`tests/unit/`)

**No optional dependencies required.** Test:
- Configuration loading/validation
- Manifest creation/serialization
- Artifact path generation
- Import error messages
- Protocol contracts (mock implementations)

**Example:**
```python
def test_training_config_rejects_negative_steps():
    with pytest.raises(ConfigError, match="steps must be"):
        TrainingConfig(steps=-100, batch_size=32, learning_rate=0.001)
```

### Integration Tests (`tests/integration/`)

**Requires optional dependencies.** Test:
- LeRobot dataset loading
- LeRobot policy loading
- LIBERO environment creation
- End-to-end training loop (1 step)
- End-to-end evaluation loop (1 episode)

**Example:**
```python
@pytest.mark.skipif(not is_available("lerobot"), reason="requires lerobot")
def test_load_lerobot_dataset():
    lerobot = require_module("lerobot", feature="the LeRobot dataset integration")
    # Integration adapter implementation here
```

## Future Architecture Considerations

### Phase A: Trajectory Quality Diagnostics

**Not yet designed.** Will add:
- Trajectory storage format (HDF5? Parquet?)
- Diagnostic protocol (`DiagnosticTool` protocol)
- Visualization protocols (`TrajectoryVisualizer` protocol)
- Integration with artifact store

**Constraint:** Must not break Phase B APIs. Manifests must be forward-compatible.

### Multi-Environment Support

**Current:** LIBERO only  
**Future:** Add protocols for:
- Real robot environments (hardware wrappers)
- Other simulators (MuJoCo, Isaac, Habitat)
- Multi-task environments

**Challenge:** Observation/action space standardization

### Multi-Policy Support

**Current:** LeRobot policies only  
**Future:** Support:
- Diffusion policies (outside LeRobot)
- VLA (vision-language-action models)
- Custom PyTorch models

**Challenge:** Checkpoint format standardization

### Distributed Training

**Current:** Single-GPU training assumed  
**Future:** Multi-GPU/multi-node via:
- PyTorch DDP
- Accelerate integration

**Challenge:** Manifest tracking across distributed runs
