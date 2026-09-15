"""
Clip-by-Clip Audio & Video Consistency Evaluator (FR-Eval).

Performs deterministic 0-token verification across individual video-audio clip pairs:
- Dialogue/subtitle extraction across source time windows
- Cross-modal activity domain conflict detection
- Dialogue-narration semantic correlation
- Temporal continuity and duration drift audit
- Structured Markdown audit table rendering
"""

from __future__ import annotations
import logging
from typing import Any

from src.eval.deterministic_eval import ACTIVITY_DOMAINS
from src.eval.models import ClipByClipReport, ClipVerificationResult
from src.media.scene_matcher import CONCEPT_MAP, build_semantic_scene_index, score_scene_for_narration, tokenize_text

logger = logging.getLogger(__name__)


def format_timestamp_mmss(seconds: float) -> str:
    """Format seconds into MM:SS format."""
    total_sec = max(0.0, float(seconds))
    mins = int(total_sec // 60)
    secs = int(total_sec % 60)
    return f"{mins:02d}:{secs:02d}"


def extract_subtitles_for_window(
    subtitles: list[dict[str, Any]] | None,
    scene_index: dict[str, Any] | None,
    source_start: float,
    source_end: float,
) -> str:
    """
    Extract all dialogue and subtitles that fall within the [source_start, source_end] window.
    """
    collected: list[str] = []

    # 1. Primary: subtitles.json records
    if subtitles:
        for sub in subtitles:
            st = float(sub.get("start", sub.get("start_seconds", 0.0)))
            et = float(sub.get("end", sub.get("end_seconds", st)))
            txt = sub.get("text", "").strip()
            if not txt:
                continue
            # Overlap check
            if st < source_end and et > source_start:
                if txt not in collected:
                    collected.append(txt)

    # 2. Secondary fallback: scene_index.json transcript_text
    if not collected and scene_index and scene_index.get("scenes"):
        for sc in scene_index["scenes"]:
            st = float(sc.get("start_seconds", 0.0))
            et = float(sc.get("end_seconds", st))
            txt = sc.get("transcript_text", "").strip()
            if not txt:
                continue
            if st < source_end and et > source_start:
                if txt not in collected:
                    collected.append(txt)

    return " ".join(collected)


def detect_activities(text: str) -> set[str]:
    """Detect activity domains present in a text snippet."""
    detected = set()
    text_lower = text.lower()
    for domain, spec in ACTIVITY_DOMAINS.items():
        # Match Chinese/English text cues
        if any(cue.lower() in text_lower for cue in spec.get("text_cues", [])):
            detected.add(domain)
        # Match scene/subtitle cues
        if any(cue.lower() in text_lower for cue in spec.get("scene_cues", [])):
            detected.add(domain)
    return detected


def verify_single_clip(
    clip_index: int,
    decision: dict[str, Any],
    subtitles: list[dict[str, Any]] | None = None,
    scene_index: dict[str, Any] | None = None,
    matching_scene: Any | None = None,
    prev_decision: dict[str, Any] | None = None,
    total_source_duration: float = 300.0,
) -> ClipVerificationResult:
    """
    Perform a deterministic verification on a single clip.
    """
    seg_id = decision.get("segment_id", f"narration-{clip_index:03d}")
    narration_text = decision.get("text", "")
    src_start = float(decision.get("source_start", 0.0))
    src_end = float(decision.get("source_end", src_start))
    duration = float(decision.get("duration", max(0.0, src_end - src_start)))
    tl_start = float(decision.get("timeline_start", 0.0))
    tl_end = float(decision.get("timeline_end", tl_start + duration))

    # Formatted strings
    recap_tl_str = f"`{format_timestamp_mmss(tl_start)} – {format_timestamp_mmss(tl_end)}` ({duration:.1f}s)"
    src_window_str = f"`{format_timestamp_mmss(src_start)} – {format_timestamp_mmss(src_end)}` ({src_start:.1f}s–{src_end:.1f}s)"

    # Extract video dialogue
    video_subs = extract_subtitles_for_window(subtitles, scene_index, src_start, src_end)

    # Activity domain conflict checking
    audio_acts = detect_activities(narration_text)
    video_acts = detect_activities(video_subs)

    findings: list[str] = []
    conflict_detected = False

    for a_act in audio_acts:
        conflicts = ACTIVITY_DOMAINS.get(a_act, {}).get("conflicts_with", [])
        for v_act in video_acts:
            if v_act in conflicts and v_act not in audio_acts:
                conflict_detected = True
                findings.append(
                    f"Cross-modal activity conflict: narration claims '{a_act}', but video contains '{v_act}' dialogue/actions"
                )

    # Chronological regression check
    if prev_decision:
        prev_src = float(prev_decision.get("source_start", 0.0))
        if src_start < prev_src - 45.0:
            findings.append(
                f"Severe backward timeline jump ({src_start:.1f}s < prev {prev_src:.1f}s)"
            )

    # Duration drift check
    v_dur = src_end - src_start
    if v_dur > 0 and duration > 0 and abs(v_dur - duration) > 1.0:
        findings.append(f"Audio-video duration drift exceeds threshold: |{v_dur:.2f}s - {duration:.2f}s| > 1.0s")

    # Semantic correlation score
    semantic_score = 100.0
    if matching_scene:
        c_score = score_scene_for_narration(
            scene=matching_scene,
            narration_text=narration_text,
            target_timestamp=src_start,
            total_duration=total_source_duration,
        )
        # Scale to 0-100
        semantic_score = round(min(100.0, max(20.0, c_score * 120.0)), 1)
        if c_score < 0.10 and not any("conflict" in f for f in findings) and len(video_subs) >= 8:
            findings.append(f"Low semantic dialogue correlation ({c_score:.2f})")
    elif video_subs and len(video_subs) >= 8:
        # Fallback keyword overlap
        narr_tokens = tokenize_text(narration_text)
        sub_tokens = tokenize_text(video_subs)
        overlap = len(narr_tokens.intersection(sub_tokens))
        ratio = overlap / max(2.0, len(narr_tokens) * 0.15)
        semantic_score = round(min(100.0, max(30.0, ratio * 100.0)), 1)
    elif not video_subs:
        semantic_score = 90.0 if not conflict_detected else 40.0

    shared_acts = audio_acts.intersection(video_acts)
    if shared_acts and not conflict_detected:
        semantic_score = max(semantic_score, 85.0)

    passed = (not conflict_detected) and (len(findings) == 0 or (len(findings) == 1 and "drift" in findings[0]))
    if conflict_detected:
        passed = False
        semantic_score = min(semantic_score, 40.0)

    # Construct verdict details
    if passed:
        if video_subs:
            v_summary = f"Consistent: matches dialogue ('{video_subs[:25]}...')" if len(video_subs) > 25 else f"Consistent: matches dialogue ('{video_subs}')"
        else:
            v_summary = "Consistent: visual framing and setting aligned"
    else:
        v_summary = f"Inconsistent: {findings[0]}" if findings else "Low alignment"

    return ClipVerificationResult(
        clip_index=clip_index,
        segment_id=seg_id,
        recap_timeline=recap_tl_str,
        source_movie_window=src_window_str,
        source_start=src_start,
        source_end=src_end,
        duration=duration,
        narration_text=narration_text,
        video_subtitles=video_subs,
        video_activity=",".join(sorted(video_acts)) if video_acts else "general",
        audio_activity=",".join(sorted(audio_acts)) if audio_acts else "general",
        semantic_match_score=semantic_score,
        passed=passed,
        findings=findings,
        verdict_details=v_summary,
    )


def verify_all_clips(
    edit_decisions: list[dict[str, Any]],
    subtitles: list[dict[str, Any]] | None = None,
    scene_index: dict[str, Any] | None = None,
    story: dict[str, Any] | None = None,
    total_source_duration: float = 300.0,
) -> ClipByClipReport:
    """
    Verify all clips in an edit plan and generate an aggregated report.
    """
    if not edit_decisions:
        return ClipByClipReport()

    semantic_idx = None
    if scene_index and scene_index.get("scenes"):
        try:
            semantic_idx = build_semantic_scene_index(scene_index, story)
        except Exception as e:
            logger.warning("Could not build semantic scene index for clip verification: %s", e)

    results: list[ClipVerificationResult] = []
    discrepancies: list[str] = []

    for i, dec in enumerate(edit_decisions):
        prev = edit_decisions[i - 1] if i > 0 else None
        src_st = float(dec.get("source_start", 0.0))

        matching_sc = None
        if semantic_idx and semantic_idx.scenes:
            matching_sc = next((s for s in semantic_idx.scenes if s.start_seconds <= src_st <= s.end_seconds), None)
            if matching_sc is None:
                matching_sc = min(semantic_idx.scenes, key=lambda s: abs(s.start_seconds - src_st))

        res = verify_single_clip(
            clip_index=i,
            decision=dec,
            subtitles=subtitles,
            scene_index=scene_index,
            matching_scene=matching_sc,
            prev_decision=prev,
            total_source_duration=total_source_duration,
        )
        results.append(res)
        if not res.passed:
            discrepancies.extend([f"Clip {i} ({res.segment_id}): {f}" for f in res.findings])

    total = len(results)
    passed_count = sum(1 for r in results if r.passed)
    failed_count = total - passed_count
    pass_rate = round((passed_count / total) * 100.0, 1) if total > 0 else 100.0
    avg_semantic = round(sum(r.semantic_match_score for r in results) / total, 1) if total > 0 else 100.0

    report = ClipByClipReport(
        total_clips=total,
        passed_clips=passed_count,
        failed_clips=failed_count,
        clip_pass_rate=pass_rate,
        average_semantic_score=avg_semantic,
        clip_results=results,
        discrepancies=discrepancies,
    )

    report.formatted_table_markdown = format_clip_verification_table(report)
    return report


def format_clip_verification_table(report: ClipByClipReport) -> str:
    """Format the clip verification results into a clean markdown table."""
    lines: list[str] = [
        "| # | Recap Timeline | Source Movie Window | On-Screen Video Content & Subtitles | Audio Narration (Voiceover) | Consistency Verdict |",
        "| :-: | :---: | :---: | :--- | :--- | :--- |",
    ]

    for r in report.clip_results:
        # Format subtitles cleanly
        subs = r.video_subtitles.replace("\n", " ").strip()
        if len(subs) > 85:
            subs = subs[:82] + "..."
        video_col = f"*{subs}*" if subs else "*(Visual scene context)*"

        # Format narration text cleanly
        narr = r.narration_text.replace("\n", " ").strip()
        if len(narr) > 90:
            narr = narr[:87] + "..."

        verdict_icon = "✅ **Pass**" if r.passed else "❌ **Fail**"
        verdict_col = f"{verdict_icon}<br>{r.verdict_details}"

        lines.append(
            f"| **{r.clip_index:02d}** | {r.recap_timeline} | {r.source_movie_window} | {video_col} | {narr} | {verdict_col} |"
        )

    return "\n".join(lines)
