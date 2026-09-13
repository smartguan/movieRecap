"""Evaluation and Anti-AI-Slop Framework."""

from __future__ import annotations
from src.eval.benchmark import BenchmarkSuiteRunner
from src.eval.models import AntiSlopReport, DimensionScore, EvalGrade
from src.eval.slop_detector import SlopDetector

__all__ = [
    "SlopDetector",
    "BenchmarkSuiteRunner",
    "AntiSlopReport",
    "DimensionScore",
    "EvalGrade",
]
