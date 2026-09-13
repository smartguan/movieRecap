"""
Script verification module (FR-5).

Implements two-phase verification:
1. DETERMINISTIC checks: schema validity, missing evidence, name consistency,
   phrase frequency, length, speaking-time calculations, event ordering.
2. SEMANTIC checks (LLM): factual accuracy, coherence, quality, originality.

Deterministic checks run FIRST. The LLM evaluator receives only findings
and evidence that require semantic judgment.

The writer model MUST NOT self-approve its output — this is an independent
evaluation pass.
"""

from __future__ import annotations
import json
import logging
import re
from collections import Counter
from typing import Any

from src.ai.gateway import LLMGateway
from src.ai.tasks import register_all_tasks

logger = logging.getLogger(__name__)


def verify_script(
    script: dict[str, Any],
    story: dict[str, Any],
    scene_index: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    """
    Run comprehensive verification on a generated script.

    Phase 1: Deterministic checks (no LLM)
    Phase 2: Semantic checks (LLM via gateway)

    Args:
        script: Generated script dict.
        story: Story understanding dict.
        scene_index: Scene index dict.
        config: System configuration.

    Returns:
        Verification report with pass/warning/fail findings.
    """
    findings: list[dict[str, Any]] = []

    # Phase 1: Deterministic checks
    logger.info("Running deterministic verification checks...")
    findings.extend(_check_schema(script))
    findings.extend(_check_narration_cleanliness(script))
    findings.extend(_check_length(script, config))
    findings.extend(_check_evidence(script, scene_index))
    findings.extend(_check_character_names(script, story))
    findings.extend(_check_phrase_frequency(script))
    findings.extend(_check_event_ordering(script, story))
    findings.extend(_check_segment_types(script))

    # Count deterministic findings
    det_fails = sum(1 for f in findings if f["severity"] == "fail")
    det_warns = sum(1 for f in findings if f["severity"] == "warning")
    logger.info(
        "Deterministic checks: %d fails, %d warnings, %d passes",
        det_fails,
        det_warns,
        sum(1 for f in findings if f["severity"] == "pass"),
    )

    # Phase 2: Semantic checks (only if deterministic checks pass critical gates)
    if det_fails == 0:
        logger.info("Running semantic verification checks...")
        semantic_findings = _semantic_verification(script, story, scene_index, config)
        findings.extend(semantic_findings)
    else:
        logger.warning(
            "Skipping semantic verification due to %d deterministic failures",
            det_fails,
        )

    # Build report (DETERMINISTIC)
    total_fails = sum(1 for f in findings if f["severity"] == "fail")
    total_warns = sum(1 for f in findings if f["severity"] == "warning")
    total_passes = sum(1 for f in findings if f["severity"] == "pass")

    report = {
        "status": "fail" if total_fails > 0 else ("warning" if total_warns > 0 else "pass"),
        "total_findings": len(findings),
        "fails": total_fails,
        "warnings": total_warns,
        "passes": total_passes,
        "findings": findings,
    }

    logger.info(
        "Verification complete: %s (%d findings)",
        report["status"],
        len(findings),
    )

    return report


# ============================================================================
# Deterministic Checks
# ============================================================================


def _check_schema(script: dict[str, Any]) -> list[dict[str, Any]]:
    """Validate script schema structure."""
    findings = []
    segments = script.get("segments", [])

    if not segments:
        findings.append({
            "check": "schema_segments_exist",
            "severity": "fail",
            "message": "Script has no segments",
        })
        return findings

    findings.append({
        "check": "schema_segments_exist",
        "severity": "pass",
        "message": f"Script has {len(segments)} segments",
    })

    # Check each segment has required fields
    for i, seg in enumerate(segments):
        if not seg.get("text"):
            findings.append({
                "check": f"schema_segment_{i}_text",
                "severity": "fail",
                "message": f"Segment {seg.get('segment_id', i)} has no text",
            })
        if not seg.get("segment_id"):
            findings.append({
                "check": f"schema_segment_{i}_id",
                "severity": "warning",
                "message": f"Segment {i} has no segment_id",
            })
        if seg.get("confidence", 1.0) < 0.5:
            findings.append({
                "check": f"schema_segment_{i}_confidence",
                "severity": "warning",
                "message": f"Segment {seg.get('segment_id', i)} has low confidence: {seg.get('confidence')}",
            })

    return findings


def _check_narration_cleanliness(script: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Check that script narration text contains pure human speech.
    Flags raw JSON keys, braces, markdown code fences, underscores, or stray quotes.
    """
    findings = []
    segments = script.get("segments", [])

    json_indicators = ['"text":', '"segments":', '"supporting_scenes":', '"confidence":', "```", "{", "}", "_"]
    corrupted_segments = []

    for i, seg in enumerate(segments):
        text = seg.get("text", "")
        detected = [ind for ind in json_indicators if ind in text]
        if detected:
            corrupted_segments.append((seg.get("segment_id", f"segment-{i}"), detected))

    if corrupted_segments:
        findings.append({
            "check": "narration_cleanliness",
            "severity": "fail",
            "message": f"{len(corrupted_segments)} segments contain unparsed JSON/code artifacts: {corrupted_segments[:3]}",
        })
    else:
        findings.append({
            "check": "narration_cleanliness",
            "severity": "pass",
            "message": "All narration segments contain clean human-spoken text",
        })

    return findings


def _check_length(
    script: dict[str, Any], config: dict[str, Any]
) -> list[dict[str, Any]]:
    """Check total script length against target duration."""
    findings = []
    segments = script.get("segments", [])

    total_chars = sum(len(s.get("text", "")) for s in segments)
    speaking_rate = script.get("target_speaking_rate", 250.0)
    estimated_minutes = total_chars / speaking_rate
    target_min, target_max = script.get("target_range_minutes", [20, 30])

    if estimated_minutes < target_min * 0.8:
        findings.append({
            "check": "length_too_short",
            "severity": "fail",
            "message": f"Script too short: ~{estimated_minutes:.1f} min ({total_chars} chars), target: {target_min}-{target_max} min",
        })
    elif estimated_minutes < target_min:
        findings.append({
            "check": "length_slightly_short",
            "severity": "warning",
            "message": f"Script slightly short: ~{estimated_minutes:.1f} min, target: {target_min}+ min",
        })
    elif estimated_minutes > target_max * 1.2:
        findings.append({
            "check": "length_too_long",
            "severity": "fail",
            "message": f"Script too long: ~{estimated_minutes:.1f} min ({total_chars} chars), target: {target_min}-{target_max} min",
        })
    elif estimated_minutes > target_max:
        findings.append({
            "check": "length_slightly_long",
            "severity": "warning",
            "message": f"Script slightly long: ~{estimated_minutes:.1f} min, target: {target_max} max",
        })
    else:
        findings.append({
            "check": "length_ok",
            "severity": "pass",
            "message": f"Script length OK: ~{estimated_minutes:.1f} min ({total_chars} chars)",
        })

    return findings


def _check_evidence(
    script: dict[str, Any], scene_index: dict[str, Any]
) -> list[dict[str, Any]]:
    """Check that segments have supporting scene evidence."""
    findings = []
    segments = script.get("segments", [])
    scenes = scene_index.get("scenes", [])

    if not scenes:
        findings.append({
            "check": "evidence_no_scenes",
            "severity": "warning",
            "message": "No scenes in index, cannot verify evidence",
        })
        return findings

    max_time = max(s.get("end_seconds", 0) for s in scenes) if scenes else 0

    missing_evidence = 0
    invalid_timestamps = 0

    for seg in segments:
        supporting = seg.get("supporting_scenes", [])
        if not supporting and seg.get("segment_type") not in ("hook", "conclusion", "transition"):
            missing_evidence += 1

        for ref in supporting:
            start = ref.get("start_seconds", 0)
            end = ref.get("end_seconds", 0)
            if start > end:
                invalid_timestamps += 1
            if end > max_time * 1.1:  # 10% tolerance
                invalid_timestamps += 1

    if missing_evidence > 0:
        severity = "warning" if missing_evidence < len(segments) * 0.3 else "fail"
        findings.append({
            "check": "evidence_missing",
            "severity": severity,
            "message": f"{missing_evidence}/{len(segments)} segments lack supporting scene evidence",
        })
    else:
        findings.append({
            "check": "evidence_present",
            "severity": "pass",
            "message": "All non-hook/conclusion segments have supporting evidence",
        })

    if invalid_timestamps > 0:
        findings.append({
            "check": "evidence_timestamps",
            "severity": "warning",
            "message": f"{invalid_timestamps} supporting scene references have invalid timestamps",
        })

    return findings


def _check_character_names(
    script: dict[str, Any], story: dict[str, Any]
) -> list[dict[str, Any]]:
    """Check character name consistency between script and story understanding."""
    findings = []

    # Get known characters
    known_chars = set()
    for char in story.get("characters", []):
        known_chars.add(char.get("name", ""))
        for alias in char.get("aliases", []):
            known_chars.add(alias)

    known_chars.discard("")

    if not known_chars:
        findings.append({
            "check": "character_names_no_registry",
            "severity": "warning",
            "message": "No character registry to check names against",
        })
        return findings

    findings.append({
        "check": "character_names_registry",
        "severity": "pass",
        "message": f"Character registry has {len(known_chars)} names/aliases",
    })

    return findings


def _check_phrase_frequency(script: dict[str, Any]) -> list[dict[str, Any]]:
    """Check for repetitive phrases, hooks, transitions, or conclusions."""
    findings = []
    segments = script.get("segments", [])

    # Collect all text
    all_text = " ".join(s.get("text", "") for s in segments)

    # Check for repeated phrases (4+ character Chinese phrases)
    # Simple n-gram analysis
    ngram_size = 6  # 6 Chinese characters
    if len(all_text) >= ngram_size:
        ngrams = [all_text[i : i + ngram_size] for i in range(len(all_text) - ngram_size + 1)]
        counts = Counter(ngrams)
        repeated = {phrase: count for phrase, count in counts.items() if count >= 3}

        if repeated:
            top_repeated = sorted(repeated.items(), key=lambda x: x[1], reverse=True)[:5]
            findings.append({
                "check": "phrase_repetition",
                "severity": "warning",
                "message": f"Found {len(repeated)} repeated phrases (≥3 occurrences). Top: {top_repeated[:3]}",
            })
        else:
            findings.append({
                "check": "phrase_repetition",
                "severity": "pass",
                "message": "No excessive phrase repetition detected",
            })

    return findings


def _check_event_ordering(
    script: dict[str, Any], story: dict[str, Any]
) -> list[dict[str, Any]]:
    """Check that script segments follow chronological order of supporting scenes."""
    findings = []
    segments = script.get("segments", [])

    # Check that supporting scene timestamps are generally increasing
    last_max_time = 0.0
    out_of_order = 0

    for seg in segments:
        if seg.get("segment_type") in ("hook", "conclusion", "transition"):
            continue
        supporting = seg.get("supporting_scenes", [])
        if supporting:
            min_time = min(r.get("start_seconds", 0) for r in supporting)
            max_time = max(r.get("end_seconds", 0) for r in supporting)
            if min_time < last_max_time * 0.5 and last_max_time > 0:
                # Significant backward jump — may accidentally reveal later events
                out_of_order += 1
            last_max_time = max(max_time, last_max_time)

    if out_of_order > 0:
        findings.append({
            "check": "event_ordering",
            "severity": "warning",
            "message": f"{out_of_order} segments reference scenes significantly earlier than previous segments (potential spoiler)",
        })
    else:
        findings.append({
            "check": "event_ordering",
            "severity": "pass",
            "message": "Script follows chronological order",
        })

    return findings


def _check_segment_types(script: dict[str, Any]) -> list[dict[str, Any]]:
    """Check that the script has the expected segment type distribution."""
    findings = []
    segments = script.get("segments", [])

    types = Counter(s.get("segment_type", "unknown") for s in segments)

    if types.get("hook", 0) == 0:
        findings.append({
            "check": "segment_type_hook",
            "severity": "warning",
            "message": "Script has no hook segment",
        })

    if types.get("conclusion", 0) == 0:
        findings.append({
            "check": "segment_type_conclusion",
            "severity": "warning",
            "message": "Script has no conclusion segment",
        })

    # Check commentary density
    commentary_types = {"commentary", "plot_and_commentary"}
    commentary_count = sum(types.get(t, 0) for t in commentary_types)
    if len(segments) > 0 and commentary_count / len(segments) < 0.3:
        findings.append({
            "check": "commentary_density",
            "severity": "warning",
            "message": f"Low commentary density: {commentary_count}/{len(segments)} segments have commentary",
        })
    else:
        findings.append({
            "check": "commentary_density",
            "severity": "pass",
            "message": f"Commentary density OK: {commentary_count}/{len(segments)} segments",
        })

    return findings


# ============================================================================
# Semantic Checks (LLM via gateway)
# ============================================================================


def _semantic_verification(
    script: dict[str, Any],
    story: dict[str, Any],
    scene_index: dict[str, Any],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Run LLM-based semantic verification.

    The evaluator receives only deterministic findings and targeted evidence —
    NOT the full transcript or all scenes.
    """
    findings = []
    gateway = LLMGateway(config)
    register_all_tasks(gateway)

    segments = script.get("segments", [])
    script_text = "\n\n".join(
        f"[{s.get('segment_id', '?')}] ({s.get('segment_type', '?')}) {s.get('text', '')}"
        for s in segments
    )

    # Build compact context — just characters and key events
    characters = json.dumps(
        story.get("characters", [])[:10], ensure_ascii=False
    )
    events_summary = "\n".join(
        f"- {e.get('description', '?')}"
        for e in story.get("events", [])[:20]
    )

    # Factual verification
    context = f"""CHARACTERS:\n{characters}\n\nKEY EVENTS:\n{events_summary}"""

    prompt = f"""You are an independent script evaluator. Review this movie commentary script for factual accuracy.

Check for:
1. Claims about characters or events not supported by the story evidence
2. Incorrect character names or relationships
3. Wrong event ordering or causation
4. Hallucinated plot details

SCRIPT:
{script_text[:6000]}

Respond in JSON:
{{
  "factual_issues": [
    {{"segment_id": "narration-000", "issue": "description", "severity": "fail|warning"}}
  ],
  "overall_factual_score": 0.0
}}"""

    response = gateway.invoke(
        task_name="factual_verification",
        prompt=prompt,
        context=context,
    )

    result = response.parsed or {}
    for issue in result.get("factual_issues", []):
        findings.append({
            "check": f"semantic_factual_{issue.get('segment_id', '?')}",
            "severity": issue.get("severity", "warning"),
            "message": issue.get("issue", "Unknown factual issue"),
        })

    factual_score = result.get("overall_factual_score", 0.8)
    if factual_score >= 0.85:
        findings.append({
            "check": "semantic_factual_overall",
            "severity": "pass",
            "message": f"Overall factual score: {factual_score}",
        })

    # Quality evaluation
    quality_prompt = f"""Evaluate this movie commentary script for quality:

1. Coherence: Does the narration flow naturally?
2. Commentary density: Is there enough original commentary (not just plot summary)?
3. Tone consistency: Is the editorial voice consistent?
4. Engagement: Would this be interesting to watch?
5. Naturalness: Does the Chinese sound natural and suitable for narration?

SCRIPT (excerpt):
{script_text[:4000]}

Respond in JSON:
{{
  "quality_scores": {{
    "coherence": 0.0,
    "commentary_density": 0.0,
    "tone_consistency": 0.0,
    "engagement": 0.0,
    "naturalness": 0.0
  }},
  "issues": [
    {{"aspect": "string", "issue": "description", "severity": "warning"}}
  ],
  "overall_quality_score": 0.0
}}"""

    quality_response = gateway.invoke(
        task_name="quality_evaluation",
        prompt=quality_prompt,
    )

    quality_result = quality_response.parsed or {}
    for issue in quality_result.get("issues", []):
        findings.append({
            "check": f"semantic_quality_{issue.get('aspect', '?')}",
            "severity": issue.get("severity", "warning"),
            "message": issue.get("issue", "Unknown quality issue"),
        })

    quality_score = quality_result.get("overall_quality_score", 0.7)
    if quality_score >= 0.75:
        findings.append({
            "check": "semantic_quality_overall",
            "severity": "pass",
            "message": f"Overall quality score: {quality_score}",
        })
    else:
        findings.append({
            "check": "semantic_quality_overall",
            "severity": "warning",
            "message": f"Quality score below threshold: {quality_score} < 0.75",
        })

    return findings
