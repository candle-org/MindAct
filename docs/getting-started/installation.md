# Installation

## Requirements

- Python ≥ 3.12
- pip or uv

## Basic Installation

Install MindAct without optional dependencies (configuration and CLI only):

```bash
git clone https://github.com/candle-org/mindact.git
cd mindact
pip install -e .
```

This installs:
- Configuration system
- CLI (`mindact` command)
- Core data structures

You can validate YAML configs without installing PyTorch or simulation environments.

## With PyTorch

To train and evaluate policies, install PyTorch:

```bash
pip install -e ".[torch]"
```

Or follow [PyTorch installation instructions](https://pytorch.org/get-started/locally/) for your platform, then:

```bash
pip install -e .
```

## With LeRobot

To use LeRobot datasets and policies:

```bash
pip install -e ".[lerobot]"
```

This installs:
- `lerobot` package from Hugging Face
- `torch` (if not already installed)
- `transformers`, `datasets`, and related dependencies

LeRobot datasets are downloaded on first use from Hugging Face Hub. Configure cache location:

```bash
export HF_HOME=/path/to/cache
```

## With LIBERO

To use LIBERO simulation environments:

```bash
pip install -e ".[libero]"
```

**LIBERO setup requirements:**

1. Install MuJoCo 2.1.0:
   ```bash
   mkdir -p ~/.mujoco
   cd ~/.mujoco
   wget https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz
   tar -xzf mujoco210-linux-x86_64.tar.gz
   ```

2. Set environment variables:
   ```bash
   export MUJOCO_PY_MUJOCO_PATH=~/.mujoco/mujoco210
   export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin
   ```

3. Install system dependencies (Ubuntu/Debian):
   ```bash
   sudo apt-get update
   sudo apt-get install -y libglew-dev patchelf libosmesa6-dev
   ```

4. Install LIBERO:
   ```bash
   pip install -e ".[libero]"
   ```

5. Download LIBERO task descriptions:
   ```bash
   python -c "from libero.libero import get_libero_path; get_libero_path('libero_90')"
   ```

See [LIBERO documentation](https://lifelong-robot-learning.github.io/LIBERO/) for troubleshooting.

## All Integrations

Install everything:

```bash
pip install -e ".[torch,lerobot,libero]"
```

## Development

For contributors:

```bash
pip install -e ".[dev]"
```

This installs:
- pytest
- ruff (linter and formatter)
- build (package build frontend)

## Verification

Test your installation:

```bash
# Check CLI is available
mindact --help

# Validate example configuration
mindact config-check configs/experiments/libero-baseline.yaml

# Run unit tests (no optional deps required)
pytest tests/unit/ -v

# Run integration tests (requires optional deps)
pytest tests/integration/ -v
```

## Platform Notes

### macOS

LIBERO is Linux-only. macOS users can:
- Use Docker for LIBERO environments
- Develop and test with LeRobot datasets/policies only
- Contribute to core framework without simulation

### Windows

Not officially supported. WSL2 recommended for LIBERO.

### GPU

PyTorch with CUDA:

```bash
# Install PyTorch with CUDA first
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Then install MindAct
pip install -e .
```

MindAct does not manage PyTorch installation. Follow PyTorch's official instructions for your hardware.
