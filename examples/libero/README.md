# LIBERO baseline example

This directory contains a compact configuration for a future LeRobot-to-LIBERO baseline. It is intentionally declarative: MindAct v0.1 validates the configuration and provides the integration boundaries, but does not yet implement the end-to-end training or rollout command.

Validate it from the repository root:

```bash
mindact config-check examples/libero/baseline.yaml
```

Install the optional simulator stack by following [the installation guide](../../docs/getting-started/installation.md). The canonical experiment configuration is kept in `configs/experiments/libero-baseline.yaml`.
