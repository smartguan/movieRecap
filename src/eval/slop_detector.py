"""
Anti-AI-Slop Quality Engine and Grading Scorer (FR-Eval).

Aggregates deterministic metrics and semantic LLM evaluations into an
actionable 0-100 quality scorecard with letter grades and CI/CD gate checks.
"""

from __future__ import annotations
import datetime
import json
import logging
from pathlib import Path
from typing import Any

from src.eval.deterministic_eval import evaluate_deterministic
from src.eval.models import AntiSlopReport, DimensionScore, EvalGrade
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

        # Audio duration from edit plan or voice files
        audio_duration = 0.0
        voice_dir = project_dir / "voice"
        if voice_dir.exists():
            # If we have voice files, we can estimate or read duration from script
            audio_duration = script.get("estimated_duration_minutes", 2.0) * 60.0

        return self.evaluate_script(
            script=script,
            story=story,
            scene_index=scene_index,
            timeline_audio_duration=audio_duration,
        )

    def evaluate_script(
        self,
        script: dict[str, Any],
        story: dict[str, Any] | None = None,
        scene_index: dict[str, Any] | None = None,
        timeline_audio_duration: float | None = None,
    ) -> AntiSlopReport:
        """
        Evaluate script content and supporting metadata.
        """
        project_id = script.get("project_id", "test-project")
        movie_title = script.get("title", "Unknown Movie")

        # 1. Deterministic Evaluation
        det_metrics, det_dims = evaluate_deterministic(
            script=script,
            scene_index=scene_index,
            timeline_audio_duration=timeline_audio_duration,
        )

        # 2. Semantic Evaluation
        sem_metrics, sem_dims = evaluate_semantic(
            script=script,
            story=story or {},
            config=self.config,
        )

        # 3. Combine all dimensions
        all_dims = det_dims + sem_dims

        # 4. Compute weighted total score
        raw_score = sum(d.score * d.weight for d in all_dims)
        total_score = min(100.0, max(0.0, round(raw_score, 1)))

        # Hard gate check: if any deterministic hard failure occurred (e.g. JSON leaks)
        is_hard_fail = len(det_metrics.hard_failures) > 0

        # Assign Grade
        if is_hard_fail or total_score < 70.0:
            grade = EvalGrade.TIER_F
            passed = False
            is_slop = True
        elif total_score >= 92.0:
            grade = EvalGrade.TIER_S
            passed = True
            is_slop = False
        elif total_score >= 85.0:
            grade = EvalGrade.TIER_A
            passed = True
            is_slop = False
        elif total_score >= 78.0:
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
            f"## 🔍 Deterministic Metric Telemetry",
            f"- **Cleanliness Score**: {report.deterministic.cleanliness_score:.1f}%",
            f"- **Lexical Diversity (TTR)**: {report.deterministic.lexical_diversity_ttr:.3f}",
            f"- **Distinct-2 Bigram Ratio**: {report.deterministic.distinct_2_grams:.3f}",
            f"- **AI Clichés Detected**: {len(report.deterministic.cliche_matches)}",
            f"- **Evidence Grounding**: {report.deterministic.evidence_grounded_ratio * 100:.1f}%",
            f"- **Chronological Consistency**: {'Passed' if report.deterministic.chronological_order_pass else 'Failed'}",
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
