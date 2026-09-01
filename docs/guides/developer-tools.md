# Developer tools

## `mindact doctor`

Run the non-invasive environment check before attempting an optional integration:

```bash
mindact doctor
```

The command reports the MindAct and Python versions, platform, and whether the
optional PyTorch, Hugging Face Hub, LeRobot, and LIBERO modules are importable.
`FOUND` means only that a module can be imported. It does not verify MuJoCo
rendering, device execution, checkpoint access, or a complete simulator setup.
Missing optional packages do not make the dependency-free core unhealthy.

## `mindact manifest show`

Inspect a run's immutable provenance without changing any artifact:

```bash
mindact manifest show outputs/run-001
# or
mindact manifest show outputs/run-001/manifest.json
```

The command prints the run ID, experiment, seed, code revision, dataset,
policy, environment, checkpoint, and evaluation artifact references. It uses
the same manifest parser as the Python API and returns a validation error for a
missing or malformed manifest.

These commands are intentionally useful in a minimal installation and do not
import optional machine-learning or simulator packages.
