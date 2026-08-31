# MindAct agent guidance

## Scope

MindAct is a PyTorch- and Hugging Face-native toolkit for reproducible embodied-policy training and evaluation. The active package lives under `src/mindact`; historical MindNLP and MindTorch code is preserved on the `legacy` branch.

## Engineering rules

- Keep core imports lightweight. Never import `torch`, `lerobot`, or `libero` at module import time; use `mindact.utils.imports.require_module()` inside integration entry points.
- Use `@runtime_checkable` protocols for integration boundaries instead of ABCs.
- Use frozen, keyword-only dataclasses for configuration and result records.
- Reject unknown configuration fields and validate values at construction time.
- Preserve experiment provenance in JSON-compatible manifests.
- Do not copy or reimplement upstream LeRobot or LIBERO internals in the core package.

## Validation

From a Python 3.12+ environment with development dependencies installed:

```bash
pytest -q
ruff check .
python -m build
```

Optional integration tests must skip cleanly when their dependencies are not installed. Do not commit checkpoints, rollout media, trajectory data, simulator caches, or experiment outputs.
