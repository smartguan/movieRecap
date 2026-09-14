"""
Stage 3: Recap Generator & Multi-Platform Exporter (FR-Stage-3).

Takes analyzed movie artifacts from Stage 2, generates proportional narration
scripts, synthesizes TTS audio, extracts video clips, renders 1080p video,
performs Anti-Slop QA, and produces platform-ready packages in output/<slug>/.
"""

from __future__ import annotations
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.ai.gateway import LLMGateway
from src.ai.script import generate_script
from src.ai.tasks import register_all_tasks
from src.ai.verify import verify_script
from src.eval.slop_detector import SlopDetector
from src.media.clip_planner import plan_and_extract_clips
from src.media.platform_exporter import PlatformExporter
from src.media.renderer import render_recap_video
from src.media.tts import generate_voice_assets

logger = logging.getLogger(__name__)


@dataclass
class Stage3Result:
    """Artifacts produced by Stage 3 (Recap Generator)."""
    slug: str
    movie_title: str
    target_recap_minutes: float
    rendered_video_path: Path
    platform_output_dir: Path
    quality_grade: str
    quality_score: float
    script_path: Path
    metadata_youtube_path: Path
    metadata_bilibili_path: Path
    subtitles_srt_path: Path
    exported_files: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "slug": self.slug,
            "movie_title": self.movie_title,
            "target_recap_minutes": self.target_recap_minutes,
            "rendered_video_path": str(self.rendered_video_path),
            "platform_output_dir": str(self.platform_output_dir),
            "quality_grade": self.quality_grade,
            "quality_score": self.quality_score,
            "script_path": str(self.script_path),
            "metadata_youtube_path": str(self.metadata_youtube_path),
            "metadata_bilibili_path": str(self.metadata_bilibili_path),
            "subtitles_srt_path": str(self.subtitles_srt_path),
            "exported_files": self.exported_files,
        }


def _resolve_stage2_dir(stage2_input: Path | str) -> Path:
    p = Path(stage2_input)
    if p.exists() and (p / "story_understanding.json").exists():
        return p

    # Check data/stages/2_analyzed/<slug>
    cand1 = Path("data/stages/2_analyzed") / str(stage2_input)
    if cand1.exists() and (cand1 / "story_understanding.json").exists():
        return cand1

    # Check data/projects/<slug>
    cand2 = Path("data/projects") / str(stage2_input)
    if cand2.exists() and (cand2 / "story_understanding.json").exists():
        return cand2

    raise FileNotFoundError(
        f"Could not find valid Stage 2 analysis package for '{stage2_input}' (missing story_understanding.json)"
    )


def _resolve_source_video(slug: str, stage2_dir: Path) -> Path:
    """Locate source.mp4 for video clip extraction."""
    # Check stage 1
    cand1 = Path("data/stages/1_download") / slug / "source.mp4"
    if cand1.exists():
        return cand1

    # Check incoming
    cand2 = Path("data/incoming") / slug / "source.mp4"
    if cand2.exists():
        return cand2

    # Check stage 2 directory itself
    if (stage2_dir / "source.mp4").exists():
        return stage2_dir / "source.mp4"

    # Check queue
    for q_sub in ("pending", "processing", "completed"):
        cand_q = Path("data/queue") / q_sub / slug / "source.mp4"
        if cand_q.exists():
            return cand_q

    raise FileNotFoundError(f"Source video (source.mp4) not found for slug '{slug}'")


def run_stage_generate(
    stage2_input: Path | str,
    output_dir: Optional[Path | str] = None,
    stage3_dir: Optional[Path | str] = None,
    target_duration: Optional[float] = None,
    duration_ratio: float = 0.20,
    config: Optional[Dict[str, Any]] = None,
    force: bool = False,
    algo_version: str = "v3",
) -> Stage3Result:
    """
    Execute Stage 3: Generate script, voiceover, video assembly, QA, and platform export.

    Args:
        stage2_input: Path to Stage 2 directory or movie slug name.
        output_dir: Base directory for platform bundles (default: output).
        stage3_dir: Working directory for Stage 3 intermediate renders (default: data/stages/3_generated/<slug>).
        target_duration: Explicit target recap duration in minutes.
        duration_ratio: Proportional duration ratio (default: 0.20 = 1/5).
        config: System configuration dict.
        force: If True, re-generate even if output exists.
        algo_version: "v3" for video-first narrative spine, "v2" for script-first (A/B testing).

    Returns:
        Stage3Result with platform package paths.
    """
    stage2_dir = _resolve_stage2_dir(stage2_input)
    slug = stage2_dir.name
    stage_work_dir = (
        Path(stage3_dir) if stage3_dir else Path("data/stages/3_generated") / slug
    )
    stage_work_dir.mkdir(parents=True, exist_ok=True)

    platform_base_dir = Path(output_dir) if output_dir else Path("output")
    platform_pkg_dir = platform_base_dir / slug
    platform_pkg_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Stage 2 Artifacts
    story_dict = json.loads((stage2_dir / "story_understanding.json").read_text(encoding="utf-8"))
    scene_index_dict = json.loads((stage2_dir / "scene_index.json").read_text(encoding="utf-8"))
    source_video_path = _resolve_source_video(slug, stage2_dir)

    movie_title = story_dict.get("movie_title") or story_dict.get("title", slug)
    source_duration_sec = scene_index_dict.get("duration_seconds", 300.0)
    source_duration_min = source_duration_sec / 60.0

    # Calculate dynamic target recap duration
    if target_duration is not None and target_duration > 0:
        target_recap_min = target_duration
    else:
        target_recap_min = max(0.5, round(source_duration_min * duration_ratio, 2))

    duration_range = [
        round(target_recap_min * 0.85, 2),
        round(target_recap_min * 1.15, 2),
    ]

    logger.info(
        "Running Stage 3 (%s) Recap Generation for '%s': Target ~%.2f min (Range: %.2f - %.2f min)",
        algo_version.upper(),
        movie_title,
        target_recap_min,
        duration_range[0],
        duration_range[1],
    )

    gateway = LLMGateway(config or {})
    register_all_tasks(gateway)

    # 2. Write Narration Script (SEMANTIC)
    script_file = stage_work_dir / "script.json"
    if not script_file.exists() or force:
        proj_data = {
            "title": movie_title,
            "project_id": slug,
            "target_duration_range": duration_range,
        }
        if algo_version.lower() == "v2":
            logger.info("Generating V2 script-first recap (%s chars target)...", int(target_recap_min * 250))
            script_dict = generate_script(
                project=proj_data,
                story=story_dict,
                scene_index=scene_index_dict,
                config=config or {},
            )
        else:
            # V3 Video-First Narrative Spine Engine
            logger.info("Generating V3 video-first narrative-spine recap (target %.2f min)...", target_recap_min)
            from src.ai.story_filter import filter_main_story_sequences
            from src.ai.video_grounded_script import VideoGroundedScriptSynthesizer
            from src.media.sequence_clusterer import cluster_scenes_into_sequences

            subtitles_file = stage2_dir / "subtitles.json"
            subtitles = (
                json.loads(subtitles_file.read_text(encoding="utf-8"))
                if subtitles_file.exists()
                else []
            )

            all_sequences = cluster_scenes_into_sequences(
                scenes=scene_index_dict.get("scenes", []),
                subtitles=subtitles,
            )

            target_budget_sec = target_recap_min * 60.0
            selected_sequences = filter_main_story_sequences(
                sequences=all_sequences,
                story_understanding=story_dict,
                target_duration_sec=target_budget_sec,
                config=config,
            )

            synthesizer = VideoGroundedScriptSynthesizer(config=config)
            script_segments = synthesizer.synthesize_script(
                title=movie_title,
                synopsis=story_dict.get("synopsis", ""),
                cast=[c.get("name", "") for c in story_dict.get("characters", []) if isinstance(c, dict)],
                genre=story_dict.get("genre", "惊悚"),
                selected_sequences=selected_sequences,
                gateway=gateway,
            )

            script_dict = {
                "title": movie_title,
                "project_id": slug,
                "version": "v3",
                "segments": script_segments,
                "target_speaking_rate": 240.0,
            }

        script_file.write_text(
            json.dumps(script_dict, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    else:
        script_dict = json.loads(script_file.read_text(encoding="utf-8"))

    # 3. Verify Script
    ver_report = verify_script(script_dict, story_dict, scene_index_dict, config or {})
    (stage_work_dir / "verification_report.json").write_text(
        json.dumps(ver_report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # 4. Generate Voice Assets (TTS)
    voice_dir = stage_work_dir / "voice_assets"
    voice_dir.mkdir(parents=True, exist_ok=True)
    voice_assets = generate_voice_assets(script_dict, voice_dir)
    (stage_work_dir / "voice_assets.json").write_text(
        json.dumps([v.model_dump() if hasattr(v, "model_dump") else v for v in voice_assets], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # 5. Plan and Extract Video Clips
    clips_dir = stage_work_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    edit_decisions = plan_and_extract_clips(
        script=script_dict,
        scene_index=scene_index_dict,
        voice_assets=voice_assets,
        source_video_path=source_video_path,
        clips_dir=clips_dir,
        story_understanding=story_dict,
    )
    (stage_work_dir / "edit_plan.json").write_text(
        json.dumps([d.model_dump() if hasattr(d, "model_dump") else d for d in edit_decisions], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # 6. Render Final 1080p Video
    renders_dir = stage_work_dir / "renders"
    renders_dir.mkdir(parents=True, exist_ok=True)
    recap_video_path = renders_dir / "recap_final.mp4"
    render_recap_video(
        edit_decisions=edit_decisions,
        output_video_path=recap_video_path,
        burn_subtitles=True,
    )

    total_audio_dur = sum(float(d.get("duration", 0.0)) for d in edit_decisions)

    # 7. Anti-Slop QA Scorecard Evaluation
    detector = SlopDetector(config=config)
    qa_report = detector.evaluate_script(
        script=script_dict,
        story=story_dict,
        scene_index=scene_index_dict,
        edit_decisions=[d.model_dump() if hasattr(d, "model_dump") else d for d in edit_decisions],
        timeline_audio_duration=total_audio_dur,
    )

    # 8. Export Multi-Platform Bundles
    assets_dir = stage_work_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    (assets_dir / "edit_decisions.json").write_text(
        json.dumps([d.model_dump() if hasattr(d, "model_dump") else d for d in edit_decisions], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    import os, shutil
    kf_stage2 = (stage2_dir / "keyframes").resolve()
    kf_work = stage_work_dir / "keyframes"
    if kf_stage2.exists() and not kf_work.exists() and not kf_work.is_symlink():
        try:
            os.symlink(kf_stage2, kf_work)
        except Exception:
            shutil.copytree(kf_stage2, kf_work, dirs_exist_ok=True)

    exporter = PlatformExporter(output_root=platform_base_dir)
    exported_dict = exporter.export_package(
        project_dir=stage_work_dir,
        movie_title=movie_title,
        metadata=story_dict,
    )
    exported_files = list(exported_dict.values())

    result = Stage3Result(
        slug=slug,
        movie_title=movie_title,
        target_recap_minutes=target_recap_min,
        rendered_video_path=recap_video_path,
        platform_output_dir=platform_pkg_dir,
        quality_grade=qa_report.grade.value,
        quality_score=qa_report.total_score,
        script_path=script_file,
        metadata_youtube_path=platform_pkg_dir / "metadata_youtube.json",
        metadata_bilibili_path=platform_pkg_dir / "metadata_bilibili.json",
        subtitles_srt_path=platform_pkg_dir / "subtitles.srt",
        exported_files=[str(p) for p in exported_files],
    )

    (stage_work_dir / "stage3_checkpoint.json").write_text(
        json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("Stage 3 complete: '%s' exported to %s (Grade %s, %.1f/100)", movie_title, platform_pkg_dir, result.quality_grade, result.quality_score)
    return result
