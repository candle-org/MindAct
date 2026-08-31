"""Evaluation interfaces and framework-neutral runners."""

from mindact.evaluation.base import EpisodeRecord, EvaluationResult, Evaluator
from mindact.evaluation.runner import EvaluationRunner

__all__ = ["EpisodeRecord", "EvaluationResult", "EvaluationRunner", "Evaluator"]


# Fake adapters are intentionally not part of the stable top-level API.
