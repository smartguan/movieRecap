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
    av_semantic_alignment_score: float = Field(100.0, ge=0.0, le=100.0)
    scene_content_match_score: float = Field(100.0, ge=0.0, le=100.0)
    low_match_segments: list[str] = Field(default_factory=list)
    phase_mismatch_count: int = 0
    coupling_score: float = Field(100.0, ge=0.0, le=100.0)
    details: str = "Audio and video clips are tightly synchronized"


class StoryContinuityMetrics(BaseModel):
    temporal_monotonicity_score: float = Field(100.0, ge=0.0, le=100.0)
    transition_coherence_ratio: float = Field(1.0, ge=0.0, le=1.0)
    character_entity_thread_ratio: float = Field(1.0, ge=0.0, le=1.0)
    visual_spine_contiguity_score: float = Field(100.0, ge=0.0, le=100.0)
    continuity_score: float = Field(100.0, ge=0.0, le=100.0)
    scene_jump_count: int = 0
    unbridged_jump_count: int = 0
    backward_jump_count: int = 0
    discontinuity_events: list[str] = Field(default_factory=list)
    details: str = "Narrative flow is strictly chronological and smoothly transitioned"


class ClipVerificationResult(BaseModel):
    clip_index: int
    segment_id: str = ""
    recap_timeline: str = ""
    source_movie_window: str = ""
    source_start: float = 0.0
    source_end: float = 0.0
    duration: float = 0.0
    narration_text: str = ""
    video_subtitles: str = ""
    video_activity: str = ""
    audio_activity: str = ""
    semantic_match_score: float = Field(100.0, ge=0.0, le=100.0)
    passed: bool = True
    findings: list[str] = Field(default_factory=list)
    verdict_details: str = "Consistent"


class ClipByClipReport(BaseModel):
    total_clips: int = 0
    passed_clips: int = 0
    failed_clips: int = 0
    clip_pass_rate: float = Field(100.0, ge=0.0, le=100.0)
    average_semantic_score: float = Field(100.0, ge=0.0, le=100.0)
    clip_results: list[ClipVerificationResult] = Field(default_factory=list)
    discrepancies: list[str] = Field(default_factory=list)
    formatted_table_markdown: str = ""


class DeterministicMetrics(BaseModel):
    cleanliness_score: float = Field(100.0, ge=0.0, le=100.0)
    movie_identity_pass: bool = True
    contaminated_entities: list[str] = Field(default_factory=list)
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
    continuity: StoryContinuityMetrics = Field(default_factory=StoryContinuityMetrics)
    clip_verification: ClipByClipReport = Field(default_factory=ClipByClipReport)


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
    clip_verification: ClipByClipReport = Field(default_factory=ClipByClipReport)
    timestamp: str

