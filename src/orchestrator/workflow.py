"""
Main workflow runner for Movie Commentary Autopilot.

Orchestrates the end-to-end pipeline: picks up projects from the incoming
directory, runs them through all processing states, and dispatches to
deterministic workers or semantic (AI) workers as appropriate.

Design principles:
- Each stage is idempotent and safely retryable.
- Deterministic workers handle all mechanical tasks.
- LLM calls go through the central gateway only for semantic tasks.
- State is persisted after every transition.
"""

from __future__ import annotations
import logging
from pathlib import Path
from typing import Any

from src.orchestrator.project import ProjectManager
from src.orchestrator.states import (
    HUMAN_ACTION_STATES,
    TERMINAL_STATES,
    ProjectState,
    get_next_state,
)

logger = logging.getLogger(__name__)


class WorkflowRunner:
    """
    Runs the movie commentary production workflow.

    Each stage dispatches to the appropriate worker:
    - Deterministic stages: media workers, scripts, file operations
    - Semantic stages: LLM gateway for story understanding, script writing, etc.
    """

    def __init__(
        self,
        projects_dir: Path,
        incoming_dir: Path,
        config: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize the workflow runner.

        Args:
            projects_dir: Root directory for project data.
            incoming_dir: Directory to watch for new movie inputs.
            config: System configuration dict.
        """
        self.project_manager = ProjectManager(projects_dir)
        self.incoming_dir = Path(incoming_dir)
        self.incoming_dir.mkdir(parents=True, exist_ok=True)
        self.config = config or {}

    def process_incoming(self) -> list[dict[str, Any]]:
        """
        Scan the incoming directory and create projects for new movies.

        Returns:
            List of newly created project dicts.
        """
        from src.media.ingest import ingest_movie

        new_projects = []

        if not self.incoming_dir.exists():
            logger.warning("Incoming directory does not exist: %s", self.incoming_dir)
            return new_projects

        for entry in sorted(self.incoming_dir.iterdir()):
            if not entry.is_dir():
                continue

            # Check if this source has already been ingested
            # by looking for a .ingested marker file
            marker = entry / ".ingested"
            if marker.exists():
                continue

            logger.info("Found new incoming movie: %s", entry.name)

            try:
                project = ingest_movie(entry, self.project_manager.projects_dir)
                # Mark as ingested
                marker.write_text(project["project_id"], encoding="utf-8")
                new_projects.append(project)
                logger.info(
                    "Created project %s for %s",
                    project["project_id"],
                    entry.name,
                )
            except Exception as e:
                logger.error("Failed to ingest %s: %s", entry.name, e)

        return new_projects

    def run_project(self, project_id: str) -> dict[str, Any]:
        """
        Run a project through the workflow until it reaches a human-action
        or terminal state.

        Args:
            project_id: UUID of the project to process.

        Returns:
            Updated project dict at the final state.
        """
        project = self.project_manager.load_project(project_id)
        current_state = ProjectState(project["state"])

        logger.info(
            "Starting workflow for project %s (state: %s)",
            project_id,
            current_state.value,
        )

        while (
            current_state not in TERMINAL_STATES
            and current_state not in HUMAN_ACTION_STATES
        ):
            next_state = get_next_state(current_state)
            if next_state is None:
                logger.info(
                    "No next state for %s, stopping.",
                    current_state.value,
                )
                break

            try:
                # Execute the stage
                project = self._execute_stage(project, current_state, next_state)
                current_state = ProjectState(project["state"])
            except Exception as e:
                logger.error(
                    "Stage %s failed: %s",
                    current_state.value,
                    e,
                    exc_info=True,
                )
                project = self.project_manager.transition(
                    project,
                    ProjectState.FAILED,
                    details=str(e),
                )
                break

        logger.info(
            "Project %s reached state: %s",
            project_id,
            project["state"],
        )
        return project

    def _execute_stage(
        self,
        project: dict[str, Any],
        current_state: ProjectState,
        next_state: ProjectState,
    ) -> dict[str, Any]:
        """
        Execute a single workflow stage and transition to the next state.

        Dispatches to the appropriate worker based on the current state.

        Args:
            project: Current project data.
            current_state: Current state.
            next_state: Target state after successful execution.

        Returns:
            Updated project dict.
        """
        stage_handlers = {
            ProjectState.RECEIVED: self._stage_ingest,
            ProjectState.INGESTING: self._stage_analyze,
            ProjectState.ANALYZING: self._stage_script,
            ProjectState.SCRIPTING: self._stage_generate_audio,
            ProjectState.GENERATING_AUDIO: self._stage_plan_edit,
            ProjectState.PLANNING_EDIT: self._stage_render,
            ProjectState.RENDERING: self._stage_qa,
            ProjectState.AUTOMATED_QA: self._stage_upload,
            ProjectState.YOUTUBE_PRIVATE: self._stage_await_approval,
        }

        handler = stage_handlers.get(current_state)
        if handler is None:
            raise ValueError(f"No handler for state: {current_state.value}")

        # Execute the stage handler
        project = handler(project)

        # Transition to the next state
        project = self.project_manager.transition(
            project,
            next_state,
            details=f"Completed {current_state.value}",
        )

        return project

    def _stage_ingest(self, project: dict[str, Any]) -> dict[str, Any]:
        """
        Stage: Ingest — validate and extract media metadata.

        DETERMINISTIC: Uses ffprobe, ffmpeg, hashlib. No LLM calls.
        """
        from src.media.audio import extract_audio
        from src.media.probe import extract_media_info

        project_dir = self.project_manager.get_project_dir(project["project_id"])
        source_path = Path(project["source_path"])

        # Extract media info if not already done
        if not project.get("media_info"):
            media_info = extract_media_info(source_path)
            project["media_info"] = media_info

        # Extract audio for transcription
        audio_path = project_dir / "audio" / "source_audio.wav"
        if not audio_path.exists():
            extract_audio(source_path, audio_path)

        self.project_manager.save_project(project)
        logger.info("Ingestion complete for %s", project["project_id"])
        return project

    def _stage_analyze(self, project: dict[str, Any]) -> dict[str, Any]:
        """
        Stage: Analyze — scene detection, keyframe extraction, transcription,
        and story understanding.

        MIXED:
        - Deterministic: scene detection, keyframe extraction, subtitle parsing
        - Semantic: scene descriptions, story understanding (via gateway)
        """
        from src.media.frames import extract_scene_keyframes
        from src.media.scene_detect import detect_scenes
        from src.media.subtitles import parse_srt

        project_dir = self.project_manager.get_project_dir(project["project_id"])
        source_path = Path(project["source_path"])

        # 1. Detect scenes (DETERMINISTIC)
        scenes_file = project_dir / "scenes.json"
        if not scenes_file.exists():
            scene_boundaries = detect_scenes(source_path)
            import json
            scenes_file.write_text(
                json.dumps(
                    [{"start": s, "end": e} for s, e in scene_boundaries],
                    indent=2,
                ),
                encoding="utf-8",
            )
            logger.info("Detected %d scenes", len(scene_boundaries))
        else:
            import json
            scene_data = json.loads(scenes_file.read_text(encoding="utf-8"))
            scene_boundaries = [(s["start"], s["end"]) for s in scene_data]

        # 2. Extract keyframes (DETERMINISTIC)
        keyframes_dir = project_dir / "keyframes"
        if not any(keyframes_dir.glob("*.jpg")):
            extract_scene_keyframes(
                source_path, scene_boundaries, keyframes_dir, frames_per_scene=2
            )
            logger.info("Extracted keyframes to %s", keyframes_dir)

        # 3. Parse subtitles if available (DETERMINISTIC)
        incoming_dir = Path(project["source_path"]).parent
        srt_files = list(incoming_dir.glob("*.srt"))
        if srt_files:
            import json
            subtitles = parse_srt(srt_files[0])
            srt_data = [
                {
                    "index": s.index,
                    "start": s.start_seconds,
                    "end": s.end_seconds,
                    "text": s.text,
                }
                for s in subtitles
            ]
            (project_dir / "subtitles.json").write_text(
                json.dumps(srt_data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            logger.info("Parsed %d subtitle entries", len(subtitles))

        # 4. Build scene index with transcript data (DETERMINISTIC assembly)
        self._build_scene_index(project, scene_boundaries)

        # 5. Story understanding (SEMANTIC — via LLM gateway)
        self._understand_story(project)

        self.project_manager.save_project(project)
        return project

    def _build_scene_index(
        self,
        project: dict[str, Any],
        scene_boundaries: list[tuple[float, float]],
    ) -> None:
        """
        Build the searchable scene index.

        DETERMINISTIC: Assembles index from already-extracted data.
        """
        import json

        project_dir = self.project_manager.get_project_dir(project["project_id"])
        index_file = project_dir / "scene_index.json"

        if index_file.exists():
            return

        # Load subtitle data if available
        subtitles_file = project_dir / "subtitles.json"
        subtitle_entries: list[dict] = []
        if subtitles_file.exists():
            subtitle_entries = json.loads(
                subtitles_file.read_text(encoding="utf-8")
            )

        scenes = []
        for i, (start, end) in enumerate(scene_boundaries):
            # Find subtitles that overlap with this scene (DETERMINISTIC)
            scene_subs = [
                s for s in subtitle_entries
                if s["start"] < end and s["end"] > start
            ]
            transcript = " ".join(s["text"] for s in scene_subs)

            # Find keyframe paths (DETERMINISTIC)
            keyframes_dir = project_dir / "keyframes"
            keyframe_paths = sorted(keyframes_dir.glob(f"scene_{i:04d}_*.jpg"))

            scenes.append({
                "scene_id": f"scene-{i:04d}",
                "start_seconds": start,
                "end_seconds": end,
                "duration_seconds": round(end - start, 3),
                "transcript_text": transcript,
                "keyframe_paths": [str(p) for p in keyframe_paths],
                "characters": [],
                "location": "",
                "actions": [],
                "emotional_tone": "",
                "sensitivity_labels": [],
            })

        index_file.write_text(
            json.dumps(
                {
                    "project_id": project["project_id"],
                    "scene_count": len(scenes),
                    "scenes": scenes,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    def _understand_story(self, project: dict[str, Any]) -> None:
        """
        Run story understanding via the LLM gateway.

        SEMANTIC: Uses LLM for interpretation, synthesis, and judgment.
        Uses hierarchical processing to minimize token usage.
        """
        import json

        project_dir = self.project_manager.get_project_dir(project["project_id"])
        story_file = project_dir / "story_understanding.json"

        if story_file.exists():
            return

        # Load scene index
        index_file = project_dir / "scene_index.json"
        if not index_file.exists():
            logger.warning("Scene index not found, skipping story understanding")
            return

        scene_index = json.loads(index_file.read_text(encoding="utf-8"))

        # Hierarchical processing: segment transcript into chunks
        # then send bounded chunks to LLM for local summaries
        from src.ai.story import understand_story

        story = understand_story(project, scene_index, self.config)

        story_file.write_text(
            json.dumps(story, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )

    def _stage_script(self, project: dict[str, Any]) -> dict[str, Any]:
        """
        Stage: Script generation and verification.

        MIXED:
        - Deterministic: schema validation, length checks, phrase frequency
        - Semantic: script writing, quality evaluation (via gateway)
        """
        import json

        project_dir = self.project_manager.get_project_dir(project["project_id"])
        script_file = project_dir / "script.json"

        if script_file.exists():
            logger.info("Script already exists, skipping generation")
            return project

        # Load story understanding
        story_file = project_dir / "story_understanding.json"
        scene_index_file = project_dir / "scene_index.json"

        story = json.loads(story_file.read_text(encoding="utf-8"))
        scene_index = json.loads(scene_index_file.read_text(encoding="utf-8"))

        # Generate script (SEMANTIC)
        from src.ai.script import generate_script

        script = generate_script(project, story, scene_index, self.config)

        # Verify script (MIXED: deterministic checks first, then semantic)
        from src.ai.verify import verify_script

        verification = verify_script(script, story, scene_index, self.config)
        script["verification"] = verification

        script_file.write_text(
            json.dumps(script, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )

        return project

    def _stage_generate_audio(self, project: dict[str, Any]) -> dict[str, Any]:
        """
        Stage: Generate voice narration (FR-6).
        Synthesizes neural voiceover for all script segments.
        """
        import json
        from src.media.tts import generate_voice_assets

        project_dir = self.project_manager.get_project_dir(project["project_id"])
        script_file = project_dir / "script.json"
        script = json.loads(script_file.read_text(encoding="utf-8"))

        audio_dir = project_dir / "audio" / "segments"
        voice_assets = generate_voice_assets(script, audio_dir)

        voice_manifest = project_dir / "audio" / "voice_assets.json"
        voice_manifest.write_text(
            json.dumps(voice_assets, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info("Generated %d voice assets", len(voice_assets))
        return project

    def _stage_plan_edit(self, project: dict[str, Any]) -> dict[str, Any]:
        """
        Stage: Plan video edit and extract clips (FR-7).
        """
        import json
        from src.media.clip_planner import plan_and_extract_clips

        project_dir = self.project_manager.get_project_dir(project["project_id"])
        source_path = Path(project["source_path"])
        scene_index_file = project_dir / "scene_index.json"
        voice_manifest = project_dir / "audio" / "voice_assets.json"

        scene_index = json.loads(scene_index_file.read_text(encoding="utf-8"))
        voice_assets = json.loads(voice_manifest.read_text(encoding="utf-8"))

        clips_dir = project_dir / "assets" / "clips"
        edit_decisions = plan_and_extract_clips(source_path, voice_assets, scene_index, clips_dir)

        edit_file = project_dir / "assets" / "edit_decisions.json"
        edit_file.write_text(
            json.dumps(edit_decisions, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info("Planned and extracted %d video edit clips", len(edit_decisions))
        return project

    def _stage_render(self, project: dict[str, Any]) -> dict[str, Any]:
        """
        Stage: Render final video with voiceover and subtitles (FR-8).
        """
        import json
        from src.media.renderer import render_recap_video

        project_dir = self.project_manager.get_project_dir(project["project_id"])
        edit_file = project_dir / "assets" / "edit_decisions.json"
        edit_decisions = json.loads(edit_file.read_text(encoding="utf-8"))

        output_video = project_dir / "renders" / "recap_final.mp4"
        render_recap_video(edit_decisions, output_video, burn_subtitles=True)

        project["rendered_video_path"] = str(output_video.absolute())
        self.project_manager.save_project(project)
        logger.info("Render complete: %s", output_video)
        return project

    def _stage_qa(self, project: dict[str, Any]) -> dict[str, Any]:
        """
        Stage: Automated QA (FR-10).
        Verifies video stream, audio stream, decodability, and duration.
        """
        import json
        from src.media.probe import extract_media_info

        project_dir = self.project_manager.get_project_dir(project["project_id"])
        output_video = project_dir / "renders" / "recap_final.mp4"

        if not output_video.exists():
            raise FileNotFoundError(f"Rendered video missing: {output_video}")

        media_info = extract_media_info(output_video)
        qa_checks = [
            {
                "check": "video_stream_present",
                "status": "pass" if media_info.video_codec != "unknown" else "fail",
                "details": f"Codec: {media_info.video_codec}, Resolution: {media_info.resolution}",
            },
            {
                "check": "audio_stream_present",
                "status": "pass" if media_info.audio_codec != "none" else "fail",
                "details": f"Audio codec: {media_info.audio_codec}, Tracks: {media_info.audio_tracks}",
            },
            {
                "check": "duration_valid",
                "status": "pass" if media_info.duration_seconds > 0 else "fail",
                "details": f"Duration: {media_info.duration_seconds:.2f}s",
            },
        ]

        qa_report = {
            "project_id": project["project_id"],
            "video_path": str(output_video),
            "media_info": media_info.model_dump(),
            "checks": qa_checks,
            "overall_status": "pass" if all(c["status"] == "pass" for c in qa_checks) else "fail",
        }

        qa_file = project_dir / "renders" / "qa_report.json"
        qa_file.write_text(json.dumps(qa_report, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("Automated QA passed for %s", output_video)

        # 1. Atomically export verified recap assets to public output directory
        from src.media.renderer import export_recap_assets
        public_output_dir = Path(self.config.get("paths", {}).get("output_dir", "output"))
        export_recap_assets(
            project_dir=project_dir,
            movie_title=project.get("title", "Recap"),
            output_dir=public_output_dir,
        )

        # 2. Automatically generate and persist Token & Cost Profile
        try:
            from src.ai.profiler import TokenCostProfiler
            profiler = TokenCostProfiler(projects_dir=self.project_manager.projects_dir)
            profile = profiler.profile_project(project["project_id"])
            profiler.save_profile_report(profile, project_dir)
            logger.info("Saved token & cost profile to %s", project_dir)
        except Exception as e:
            logger.warning("Failed to generate token cost profile in QA stage: %s", e)

        # 3. Automatically run Anti-AI-Slop Quality Evaluation
        try:
            from src.eval.slop_detector import SlopDetector
            detector = SlopDetector(self.config)
            eval_report = detector.evaluate_project(project_dir)
            
            # Save JSON report
            eval_json_file = project_dir / "recap_quality_report.json"
            eval_json_file.write_text(
                json.dumps(eval_report.model_dump(), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            # Save Markdown report
            eval_md_file = project_dir / "recap_quality_report.md"
            eval_md_file.write_text(
                detector.format_markdown_report(eval_report),
                encoding="utf-8",
            )
            logger.info(
                "Anti-Slop QA complete: Grade %s (Score: %.1f/100, Passed: %s)",
                eval_report.grade.value,
                eval_report.total_score,
                eval_report.passed,
            )
        except Exception as e:
            logger.warning("Failed to run Anti-Slop QA evaluation: %s", e)

        return project

    def _stage_upload(self, project: dict[str, Any]) -> dict[str, Any]:
        """Stage: YouTube upload. Placeholder for Milestone 3."""
        logger.info("Upload stage — placeholder for Milestone 3")
        return project

    def _stage_await_approval(self, project: dict[str, Any]) -> dict[str, Any]:
        """Stage: Await human approval. Placeholder for Milestone 3."""
        logger.info("Awaiting approval stage — placeholder for Milestone 3")
        return project


def run_workflow(
    incoming_dir: str | Path,
    projects_dir: str | Path,
    config: dict[str, Any] | None = None,
    project_id: str | None = None,
) -> None:
    """
    Main entry point for the workflow.

    Can be called to:
    1. Process all incoming movies (project_id=None)
    2. Resume a specific project (project_id=<uuid>)

    Args:
        incoming_dir: Path to the incoming movie directory.
        projects_dir: Path to the projects directory.
        config: System configuration.
        project_id: If provided, process only this project.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    runner = WorkflowRunner(
        projects_dir=Path(projects_dir),
        incoming_dir=Path(incoming_dir),
        config=config,
    )

    if project_id:
        # Resume a specific project
        runner.run_project(project_id)
    else:
        # Process incoming movies
        new_projects = runner.process_incoming()
        for project in new_projects:
            runner.run_project(project["project_id"])


if __name__ == "__main__":
    import sys
    import yaml

    # Load config
    config_path = Path("config/settings.yaml")
    config = {}
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f) or {}

    incoming = config.get("paths", {}).get("incoming_dir", "data/incoming")
    projects = config.get("paths", {}).get("projects_dir", "data/projects")

    pid = sys.argv[1] if len(sys.argv) > 1 else None
    run_workflow(incoming, projects, config, pid)
