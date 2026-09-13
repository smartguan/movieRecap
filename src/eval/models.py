"""
Data models for Evaluation and Anti-AI-Slop Framework.
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
    timestamp: str
