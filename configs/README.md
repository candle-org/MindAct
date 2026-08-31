# Experiment configurations

Experiment YAML files are versioned inputs to MindAct runs. They identify the
LeRobot dataset, policy, simulator task suite, seed, training controls and
evaluation controls. Keep machine-specific paths in `options` or local files,
not in shared examples.

Validate a file without launching training:

```bash
mindact config-check configs/experiments/libero-baseline.yaml
```
