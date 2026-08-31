# Benchmarks

This directory contains reproducible benchmark entry points for MindAct. Benchmark implementations should remain thin: dataset loading, policy execution, and simulator details belong in the corresponding adapters or integrations.

## Planned benchmark matrix

| Benchmark | Dataset | Environment | Status |
|---|---|---|---|
| LIBERO spatial baseline | LeRobot-compatible demonstrations | LIBERO spatial suite | Planned |
| LIBERO object baseline | LeRobot-compatible demonstrations | LIBERO object suite | Planned |
| Policy throughput | Synthetic observations | No simulator | Planned |

## Benchmark requirements

Every benchmark should:

1. Accept a checked-in YAML configuration or an explicit equivalent.
2. Record the exact dataset, policy, environment, and code revisions.
3. Write a manifest before executing rollouts or training.
4. Report aggregate metrics as JSON in the experiment artifact directory.
5. Make seed, task selection, episode count, and maximum episode length explicit.
6. Document hardware, software versions, and optional dependencies.

Do not commit checkpoints, videos, trajectory data, simulator caches, or generated reports. Store those outputs outside the source tree or in an ignored artifact directory.

The v0.1 release provides the schemas and artifact primitives needed by benchmark runners; executable benchmark commands will be added with the first concrete integration.
