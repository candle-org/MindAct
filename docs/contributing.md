# Contributing

MindAct welcomes focused contributions that improve reproducible embodied-policy research. The project is in an early foundation phase, so small, well-tested changes are easier to review and integrate.

## Development setup

MindAct requires Python 3.12 or newer:

```bash
git clone https://github.com/candle-org/mindact.git
cd mindact
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

Install an integration extra only when working on that integration:

```bash
pip install -e ".[lerobot]"
pip install -e ".[libero]"
```

## Before opening a pull request

Run the local quality checks:

```bash
pytest -q
ruff check .
python -m build
```

Also check the basic public paths:

```bash
python -c "import mindact; print(mindact.__version__)"
mindact --help
mindact config-check configs/experiments/libero-baseline.yaml
```

Tests that need LeRobot or LIBERO should be marked with the corresponding pytest marker and skip cleanly when the optional dependency is unavailable.

## Design guidelines

### Keep the core lightweight

Do not import PyTorch, LeRobot, LIBERO, or other heavyweight integrations at module import time. Use `mindact.utils.imports.require_module` inside an explicitly requested integration path. Configuration, manifest, artifact, and protocol tests must run without optional integrations.

### Use protocols at integration boundaries

New dataset, policy, environment, trainer, and evaluator boundaries should use `@runtime_checkable` protocols from `typing`. Prefer thin adapters over copying upstream implementations or requiring external classes to inherit from MindAct classes.

### Preserve provenance

Any new run- or evaluation-producing component should record the identifiers and revisions needed to reproduce its inputs. Keep manifests JSON-compatible and avoid putting large rollout arrays into them.

### Keep schemas strict

Configuration fields should be validated at construction time. Reject unknown keys when loading YAML so spelling mistakes cannot silently change an experiment.

## Pull requests

Use a focused branch and describe:

- The problem and intended user-facing behavior
- The public API or configuration changes
- Tests added or updated
- Optional dependencies needed to exercise the change
- Any known platform limitations

Do not include generated checkpoints, rollout videos, simulator caches, or experiment outputs in a pull request. The repository `.gitignore` lists the expected artifact patterns.

## Repository history

The repository was previously MindNLP. That codebase is retained on the `legacy` branch for historical continuity; new MindAct code should not add compatibility imports or MindSpore-specific behavior to the active package.

## License

Contributions are accepted under the [Apache License 2.0](../LICENSE). By submitting a contribution, you agree that it may be distributed under that license.
