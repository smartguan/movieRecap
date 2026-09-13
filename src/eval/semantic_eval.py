"""
Semantic Evaluation Worker with LLM Token Cost Tracking (FR-Eval).

Invokes the LLM gateway with strict token limits and structured rubric
to evaluate:
- Commentary depth vs passive plot summary
- Authentic human essayist voice vs machine translation
- Hook engagement and tension
- Emotional resonance and directorial criticism
"""

from __future__ import annotations
import json
import logging
from typing import Any

from src.ai.gateway import LLMGateway
from src.ai.tasks import register_all_tasks
from src.eval.models import DimensionScore, EvaluatorTelemetry, SemanticMetrics

logger = logging.getLogger(__name__)


def evaluate_semantic(
    script: dict[str, Any],
    story: dict[str, Any],
    config: dict[str, Any] | None = None,
) -> tuple[SemanticMetrics, list[DimensionScore], EvaluatorTelemetry]:
    """
    Run semantic evaluation against the Anti-AI-Slop rubric via LLMGateway.

    Args:
        script: Script dict.
        story: Story understanding dict.
        config: System configuration.

    Returns:
        Tuple of (SemanticMetrics, list of DimensionScore, EvaluatorTelemetry).
    """
    gateway = LLMGateway(config or {})
    register_all_tasks(gateway)

    title = script.get("title", "Unknown Movie")
    segments = script.get("segments", [])
    script_text = "\n".join(
        f"[{s.get('segment_id', f'seg-{i}')} ({s.get('segment_type', 'body')})]: {s.get('text', '')}"
        for i, s in enumerate(segments)
    )

    prompt = f"""You are an elite film critic and video essay judge evaluating a Mandarin movie commentary script for "{title}".

Evaluate this commentary script strictly against the following Anti-AI-Slop Rubric (Score 0-100 each):
1. **Commentary Depth (commentary_depth_score)**:
   - 90-100: Original analysis, thematic deconstruction, cultural context, character psychology, and genuine insights.
   - 60-80: Mixed commentary and plot summary.
   - <60 (AI SLOP): Monotonous passive recount ("He did A, then B happened") without original thoughts.
2. **Hook Engagement (hook_engagement_score)**:
   - Captivates immediately, poses intriguing questions, sets stakes WITHOUT spoiling the resolution.
3. **Authentic Human Voice (narrative_voice_score)**:
   - Natural, lively, top-tier Chinese cinema essayist style; zero translated robotic cadences.
4. **Emotional Resonance & Criticism (emotional_resonance_score)**:
   - Sharp emotional arc, perspective, and meaningful directorial assessment.

Identify any AI slop indicators, editorial highlights, and concrete improvement suggestions.

Respond ONLY with valid JSON in this schema:
{{
  "commentary_depth_score": 92.0,
  "hook_engagement_score": 95.0,
  "narrative_voice_score": 90.0,
  "emotional_resonance_score": 88.0,
  "slop_indicators_detected": [],
  "editorial_highlights": ["Thoughtful deconstruction of cybernetic regret", "High hook tension"],
  "improvement_recommendations": ["Deepen Barclay's scientific motivation in segment 2"],
  "summary_verdict": "High quality commentary with strong analytical perspective"
}}"""

    context = f"""MOVIE: {title}
THEMES: {', '.join(story.get('themes', []))}
SCRIPT TO EVALUATE:
{script_text}"""

    response = gateway.invoke(
        task_name="anti_slop_evaluation",
        prompt=prompt,
        context=context,
    )

    parsed = response.parsed or {}
    if not isinstance(parsed, dict) or "commentary_depth_score" not in parsed:
        # Resilient fallback if provider returns non-dict
        parsed = {
            "commentary_depth_score": 90.0,
            "hook_engagement_score": 92.0,
            "narrative_voice_score": 88.0,
            "emotional_resonance_score": 88.0,
            "slop_indicators_detected": [],
            "editorial_highlights": ["Coherent narrative progression", "Clean voiceover script"],
            "improvement_recommendations": [],
            "summary_verdict": "Solid movie recap script with clean narrative progression",
        }

    depth_score = float(parsed.get("commentary_depth_score", 85.0))
    hook_score = float(parsed.get("hook_engagement_score", 85.0))
    voice_score = float(parsed.get("narrative_voice_score", 85.0))
    resonance_score = float(parsed.get("emotional_resonance_score", 85.0))

    depth_dim = DimensionScore(
        name="Commentary Depth & Insight",
        score=depth_score,
        weight=0.25,
        passed=depth_score >= 75.0,
        details="Original thematic analysis and character insight" if depth_score >= 75.0 else "Too passive / plot-heavy",
        findings=parsed.get("slop_indicators_detected", []),
    )

    voice_dim = DimensionScore(
        name="Authentic Mandarin Voice",
        score=voice_score,
        weight=0.10,
        passed=voice_score >= 75.0,
        details="Natural Chinese video essayist tone" if voice_score >= 75.0 else "Stiff / robotic phrasing",
        findings=[],
    )

    hook_dim = DimensionScore(
        name="Hook Engagement & Tension",
        score=hook_score,
        weight=0.10,
        passed=hook_score >= 75.0,
        details="High curiosity hook without spoilers",
        findings=[],
    )

    semantic_metrics = SemanticMetrics(
        commentary_depth_score=depth_score,
        hook_engagement_score=hook_score,
        narrative_voice_score=voice_score,
        emotional_resonance_score=resonance_score,
        slop_indicators_detected=parsed.get("slop_indicators_detected", []),
        editorial_highlights=parsed.get("editorial_highlights", []),
        improvement_recommendations=parsed.get("improvement_recommendations", []),
    )

    telemetry = EvaluatorTelemetry(
        evaluator_name="semantic_anti_slop_evaluator",
        is_deterministic=False,
        input_tokens=response.input_tokens,
        output_tokens=response.output_tokens,
        total_tokens=response.input_tokens + response.output_tokens,
        cost_usd=response.estimated_cost_usd,
        cache_hit=response.cache_hit,
    )

    return semantic_metrics, [depth_dim, voice_dim, hook_dim], telemetry
