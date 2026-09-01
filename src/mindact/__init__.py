"""MindAct: reproducible training and evaluation for embodied policies."""

__version__ = "0.1.0"

from mindact.configs import (
    ConfigError,
    DatasetConfig,
    EnvironmentConfig,
    EvaluationConfig,
    ExperimentConfig,
    PolicyConfig,
    TrainingConfig,
)
from mindact.diagnostics import DiagnosticCheck, DoctorReport, run_doctor
from mindact.evaluation import EpisodeRecord, EvaluationResult, EvaluationRunner, Evaluator
from mindact.experiments import ArtifactStore, ExperimentManifest

__all__ = [
    "ArtifactStore",
    "DiagnosticCheck",
    "DoctorReport",
    "EpisodeRecord",
    "EvaluationResult",
    "EvaluationRunner",
    "Evaluator",
    "ConfigError",
    "DatasetConfig",
    "EnvironmentConfig",
    "EvaluationConfig",
    "ExperimentConfig",
    "ExperimentManifest",
    "PolicyConfig",
    "TrainingConfig",
    "__version__",
    "run_doctor",
]
