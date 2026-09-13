"""
Data models for Evaluation, Cross-Modal AV Coupling, and Token Profiling Framework.
"""

from __future__ import annotations
from enum import Enum
from typing import Any
from pydantic import BaseModel, Field


class EvalGrade(str, Enum):
    TIER_S = "S"
    TIER_A = "A"
    TIER_B = "B"
    TIER_C = "C"
    TIER_F = "F"


class DimensionScore(BaseModel):
    name: str
    score: float = Field(..., ge=0.0, le=100.0)
    weight: float = Field(..., ge=0.0, le=1.0)
    passed: bool
    details: str
    findings: list[str] = Field(default_factory=list)


class EvaluatorTelemetry(BaseModel):
    evaluator_name: str
    is_deterministic: bool
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    cache_hit: bool = False


class AudioVideoCouplingMetrics(BaseModel):
    av_duration_drift_seconds: float = 0.0
    character_visual_alignment_ratio: float = 1.0
    max_shot_duration_seconds: float = 0.0
    dynamic_pacing_pass: bool = True
    coupling_score: float = Field(100.0, ge=0.0, le=100.0)
    details: str = "Audio and video clips are tightly synchronized"


class DeterministicMetrics(BaseModel):
    cleanliness_score: float = Field(100.0, ge=0.0, le=100.0)
    lexical_diversity_ttr: float = 0.0
    distinct_2_grams: float = 0.0
    distinct_3_grams: float = 0.0
    cliche_matches: list[str] = Field(default_factory=list)
    repeated_phrases: list[tuple[str, int]] = Field(default_factory=list)
    speaking_rate_chars_per_min: float = 0.0
    evidence_grounded_ratio: float = 0.0
    chronological_order_pass: bool = True
    hard_failures: list[str] = Field(default_factory=list)
    av_coupling: AudioVideoCouplingMetrics = Field(default_factory=AudioVideoCouplingMetrics)


class SemanticMetrics(BaseModel):
    commentary_depth_score: float = Field(..., ge=0.0, le=100.0)
    hook_engagement_score: float = Field(..., ge=0.0, le=100.0)
    narrative_voice_score: float = Field(..., ge=0.0, le=100.0)
    emotional_resonance_score: float = Field(..., ge=0.0, le=100.0)
    slop_indicators_detected: list[str] = Field(default_factory=list)
    editorial_highlights: list[str] = Field(default_factory=list)
    improvement_recommendations: list[str] = Field(default_factory=list)


class AntiSlopReport(BaseModel):
    project_id: str
    movie_title: str
    total_score: float = Field(..., ge=0.0, le=100.0)
    grade: EvalGrade
    passed: bool
    is_slop: bool
    deterministic: DeterministicMetrics
    semantic: SemanticMetrics
    dimensions: list[DimensionScore]
    summary_verdict: str
    eval_telemetry: list[EvaluatorTelemetry] = Field(default_factory=list)
    total_eval_tokens: int = 0
    total_eval_cost_usd: float = 0.0
    deterministic_savings_description: str = "Deterministic evaluators executed with 0 LLM tokens"
    timestamp: str
