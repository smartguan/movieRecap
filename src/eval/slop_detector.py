"""
Anti-AI-Slop Quality Engine and Grading Scorer (FR-Eval).

Aggregates deterministic metrics, cross-modal AV coupling, and semantic LLM
evaluations with detailed per-evaluator token cost telemetry.
"""

from __future__ import annotations
import datetime
import json
import logging
from pathlib import Path
from typing import Any

from src.eval.deterministic_eval import evaluate_deterministic
from src.eval.models import AntiSlopReport, DimensionScore, EvalGrade, EvaluatorTelemetry
from src.eval.semantic_eval import evaluate_semantic

logger = logging.getLogger(__name__)


class SlopDetector:
    """
    Evaluator that audits movie commentary scripts and rendered assets
    to prevent shipping AI Slop to viewers.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}

    def evaluate_project(
        self,
        project_dir: Path,
    ) -> AntiSlopReport:
        """
        Evaluate a complete project directory.

        Args:
            project_dir: Path to project containing script.json, scene_index.json, etc.

        Returns:
            AntiSlopReport with complete metric breakdown and grade.
        """
        project_dir = Path(project_dir)
        script_file = project_dir / "script.json"
        if not script_file.exists():
            raise FileNotFoundError(f"Missing script.json in {project_dir}")

        script = json.loads(script_file.read_text(encoding="utf-8"))

        story_file = project_dir / "story_understanding.json"
        story = json.loads(story_file.read_text(encoding="utf-8")) if story_file.exists() else {}

        scene_index_file = project_dir / "scene_index.json"
        scene_index = json.loads(scene_index_file.read_text(encoding="utf-8")) if scene_index_file.exists() else {}

        edit_plan_file = project_dir / "edit_plan.json"
        edit_decisions = json.loads(edit_plan_file.read_text(encoding="utf-8")) if edit_plan_file.exists() else []

        audio_duration = script.get("estimated_duration_minutes", 2.0) * 60.0

        return self.evaluate_script(
            script=script,
            story=story,
            scene_index=scene_index,
            edit_decisions=edit_decisions,
            timeline_audio_duration=audio_duration,
        )

    def evaluate_script(
        self,
        script: dict[str, Any],
        story: dict[str, Any] | None = None,
        scene_index: dict[str, Any] | None = None,
        edit_decisions: list[dict[str, Any]] | None = None,
        timeline_audio_duration: float | None = None,
    ) -> AntiSlopReport:
        """
        Evaluate script content and supporting metadata.
        """
        project_id = script.get("project_id", "test-project")
        movie_title = script.get("title", "Unknown Movie")
        eval_telemetry: list[EvaluatorTelemetry] = []

        # 1. Deterministic Evaluation (0 Tokens)
        det_metrics, det_dims, det_telemetry = evaluate_deterministic(
            script=script,
            story=story,
            scene_index=scene_index,
            edit_decisions=edit_decisions,
            timeline_audio_duration=timeline_audio_duration,
        )
        eval_telemetry.append(det_telemetry)

        # 2. Semantic Evaluation
        # If deterministic checks had hard failures (e.g. JSON leaks), skip semantic calls to save tokens!
        is_hard_fail = len(det_metrics.hard_failures) > 0

        if not is_hard_fail:
            sem_metrics, sem_dims, sem_telemetry = evaluate_semantic(
                script=script,
                story=story or {},
                config=self.config,
            )
            eval_telemetry.append(sem_telemetry)
        else:
            logger.warning("Skipping semantic evaluation due to deterministic hard failures (saved 100% eval tokens)")
            from src.eval.models import SemanticMetrics
            sem_metrics = SemanticMetrics(
                commentary_depth_score=0.0,
                hook_engagement_score=0.0,
                narrative_voice_score=0.0,
                emotional_resonance_score=0.0,
                slop_indicators_detected=["Deterministic hard failure present"],
            )
            sem_dims = [
                DimensionScore(name="Commentary Depth & Insight", score=0.0, weight=0.25, passed=False, details="Skipped due to hard failure"),
                DimensionScore(name="Authentic Mandarin Voice", score=0.0, weight=0.10, passed=False, details="Skipped due to hard failure"),
                DimensionScore(name="Hook Engagement & Tension", score=0.0, weight=0.10, passed=False, details="Skipped due to hard failure"),
            ]

        # 3. Combine all dimensions
        all_dims = det_dims + sem_dims

        # 4. Compute weighted total score
        raw_score = sum(d.score * d.weight for d in all_dims)
        total_score = min(100.0, max(0.0, round(raw_score, 1)))

        # Assign Grade
        if is_hard_fail or total_score < 70.0:
            grade = EvalGrade.TIER_F
            passed = False
            is_slop = True
        elif total_score >= 90.0:
            grade = EvalGrade.TIER_S
            passed = True
            is_slop = False
        elif total_score >= 82.0:
            grade = EvalGrade.TIER_A
            passed = True
            is_slop = False
        elif total_score >= 75.0:
            grade = EvalGrade.TIER_B
            passed = True
            is_slop = False
        else:
            grade = EvalGrade.TIER_C
            passed = False
            is_slop = True

        # Generate summary verdict
        if is_hard_fail:
            verdict = f"FAILED: Hard deterministic failure detected ({det_metrics.hard_failures[0]})"
        elif is_slop:
            verdict = f"REJECTED: Quality score ({total_score:.1f}/100) indicates AI Slop risk"
        else:
            verdict = f"APPROVED (Grade {grade.value}): High quality authentic commentary ({total_score:.1f}/100)"

        total_tokens = sum(t.total_tokens for t in eval_telemetry)
        total_cost = sum(t.cost_usd for t in eval_telemetry)

        report = AntiSlopReport(
            project_id=project_id,
            movie_title=movie_title,
            total_score=total_score,
            grade=grade,
            passed=passed,
            is_slop=is_slop,
            deterministic=det_metrics,
            semantic=sem_metrics,
            dimensions=all_dims,
            summary_verdict=verdict,
            eval_telemetry=eval_telemetry,
            total_eval_tokens=total_tokens,
            total_eval_cost_usd=round(total_cost, 6),
            deterministic_savings_description=(
                f"5 deterministic evaluators (Cleanliness, Diversity, Evidence, AV Coupling, Pacing) executed at 0 tokens ($0.00)"
            ),
            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        )

        return report

    def format_markdown_report(self, report: AntiSlopReport) -> str:
        """Format an AntiSlopReport into clean GitHub-flavored markdown."""
        status_emoji = "✅ PASSED" if report.passed else "❌ FAILED / AI SLOP"

        md = [
            f"# 🎬 Movie Recap Quality & Anti-Slop Scorecard: {report.movie_title}",
            f"",
            f"- **Overall Grade**: **Tier {report.grade.value}** ({report.total_score:.1f} / 100.0)",
            f"- **Status**: **{status_emoji}**",
            f"- **Verdict**: {report.summary_verdict}",
            f"- **Project ID**: `{report.project_id}`",
            f"",
            f"## 📊 Dimensional Quality Breakdown",
            f"",
            f"| Dimension | Weight | Score | Status | Details |",
            f"| :--- | :---: | :---: | :---: | :--- |",
        ]

        for dim in report.dimensions:
            dim_status = "✅ Pass" if dim.passed else "⚠️ Warning/Fail"
            md.append(f"| **{dim.name}** | {dim.weight * 100:.0f}% | {dim.score:.1f} | {dim_status} | {dim.details} |")

        md.extend([
            f"",
            f"## 🎞 Cross-Modal Audio-Visual Coupling Telemetry",
            f"- **Audio-Video Duration Drift**: `{report.deterministic.av_coupling.av_duration_drift_seconds:.3f}s`",
            f"- **Character Visual Alignment**: `{report.deterministic.av_coupling.character_visual_alignment_ratio * 100:.1f}%`",
            f"- **Max Shot Duration**: `{report.deterministic.av_coupling.max_shot_duration_seconds:.1f}s` (Pacing pass: `{report.deterministic.av_coupling.dynamic_pacing_pass}`)",
            f"- **Coupling Status**: {report.deterministic.av_coupling.details}",
            f"",
            f"## 💰 Evaluator LLM Token & Cost Telemetry",
            f"- **Total Evaluator Tokens**: **{report.total_eval_tokens}**",
            f"- **Total Evaluator Cost**: **${report.total_eval_cost_usd:.6f} USD**",
            f"- **Deterministic Offload**: {report.deterministic_savings_description}",
            f"",
            f"| Evaluator Task | Type | In Tokens | Out Tokens | Total Tokens | Cost (USD) | Cache Hit |",
            f"| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
        ])

        for t in report.eval_telemetry:
            eval_type = "Deterministic (0 tok)" if t.is_deterministic else "Semantic LLM"
            md.append(
                f"| **{t.evaluator_name}** | {eval_type} | {t.input_tokens} | {t.output_tokens} | {t.total_tokens} | ${t.cost_usd:.6f} | {t.cache_hit} |"
            )

        md.extend([
            f"",
            f"## 💡 Semantic Review & Editorial Notes",
        ])

        if report.semantic.editorial_highlights:
            md.append("### ✨ Highlights:")
            for h in report.semantic.editorial_highlights:
                md.append(f"- {h}")

        if report.semantic.improvement_recommendations:
            md.append("\n### 🛠 Improvement Recommendations:")
            for r in report.semantic.improvement_recommendations:
                md.append(f"- {r}")

        return "\n".join(md)
