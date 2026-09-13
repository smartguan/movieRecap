"""
Benchmark Test Suite Runner for Movie Commentary Quality (FR-Eval).

Executes end-to-end evaluation across standard test movie fixtures,
tracking quality scores, token costs, regression rates, and anti-slop pass rates.
"""

from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Any

from src.ai.profiler import TokenCostProfiler
from src.eval.models import AntiSlopReport
from src.eval.slop_detector import SlopDetector
from src.orchestrator.project import ProjectManager

logger = logging.getLogger(__name__)


class BenchmarkSuiteRunner:
    """
    Automated benchmark test harness for movie recap quality regression testing.
    """

    def __init__(
        self,
        projects_dir: Path,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.projects_dir = Path(projects_dir)
        self.project_manager = ProjectManager(self.projects_dir)
        self.config = config or {}
        self.detector = SlopDetector(self.config)
        self.profiler = TokenCostProfiler(self.projects_dir)

    def run_suite(self) -> dict[str, Any]:
        """
        Run evaluation across all available projects in data/projects.

        Returns:
            Dict containing benchmark summary, per-project scorecards, and regression status.
        """
        projects = self.project_manager.list_projects()
        results: list[dict[str, Any]] = []

        total_tested = 0
        total_passed = 0
        slop_count = 0
        total_tokens = 0
        total_cost_usd = 0.0

        for p in projects:
            p_id = p["project_id"]
            p_dir = self.project_manager.get_project_dir(p_id)
            script_file = p_dir / "script.json"

            if not script_file.exists():
                logger.warning("Project %s has no script.json, skipping benchmark eval", p_id)
                continue

            total_tested += 1
            try:
                report = self.detector.evaluate_project(p_dir)
                if report.passed:
                    total_passed += 1
                if report.is_slop:
                    slop_count += 1

                # Profile token cost if telemetry exists
                cost_report = None
                try:
                    cost_report = self.profiler.profile_project(p_id)
                    total_tokens += cost_report.total_tokens
                    total_cost_usd += cost_report.total_cost_usd
                except Exception:
                    pass

                results.append({
                    "project_id": p_id,
                    "title": report.movie_title,
                    "score": report.total_score,
                    "grade": report.grade.value,
                    "passed": report.passed,
                    "is_slop": report.is_slop,
                    "tokens": cost_report.total_tokens if cost_report else 0,
                    "cost_usd": cost_report.total_cost_usd if cost_report else 0.0,
                    "verdict": report.summary_verdict,
                    "report": report.model_dump(),
                })
            except Exception as e:
                logger.error("Failed to evaluate benchmark project %s: %s", p_id, e)

        pass_rate = (total_passed / total_tested * 100.0) if total_tested > 0 else 0.0
        avg_score = (
            sum(r["score"] for r in results) / len(results)
        ) if results else 0.0

        benchmark_summary = {
            "total_movies_evaluated": total_tested,
            "movies_passed": total_passed,
            "slop_detected_count": slop_count,
            "pass_rate_percent": round(pass_rate, 1),
            "average_quality_score": round(avg_score, 1),
            "total_tokens_consumed": total_tokens,
            "total_cost_usd": round(total_cost_usd, 4),
            "results": results,
        }

        return benchmark_summary

    def format_benchmark_table(self, summary: dict[str, Any]) -> str:
        """Format benchmark suite results into a markdown scorecard table."""
        lines = [
            f"# 🏆 Movie Recap Quality & Anti-Slop Benchmark Suite",
            f"",
            f"- **Evaluated Recaps**: {summary['total_movies_evaluated']}",
            f"- **Pass Rate**: **{summary['pass_rate_percent']:.1f}%** ({summary['movies_passed']}/{summary['total_movies_evaluated']})",
            f"- **Average Quality Score**: **{summary['average_quality_score']:.1f} / 100.0**",
            f"- **Total LLM Cost**: ${summary['total_cost_usd']:.4f} USD ({summary['total_tokens_consumed']} tokens)",
            f"",
            f"## 🎬 Test Movie Quality Scorecards",
            f"",
            f"| Movie Title | Grade | Score | Status | Tokens | Cost (USD) | Summary Verdict |",
            f"| :--- | :---: | :---: | :---: | :---: | :---: | :--- |",
        ]

        for r in summary.get("results", []):
            status = "✅ Pass" if r["passed"] else "❌ Slop/Fail"
            lines.append(
                f"| **{r['title']}** | **{r['grade']}** | {r['score']:.1f} | {status} | {r['tokens']} | ${r['cost_usd']:.4f} | {r['verdict']} |"
            )

        return "\n".join(lines)
