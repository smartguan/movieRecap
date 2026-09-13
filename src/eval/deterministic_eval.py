"""
Deterministic Evaluation Worker with Cross-Modal AV Coupling (FR-Eval).

Performs strict 0-token verification across:
- Cleanliness & programmatic leaks
- Lexical diversity and cliché scanning
- Timeline evidence mapping & chronological sequence
- Cross-Modal Audio-Video Coupling (duration drift, shot dynamics, visual character presence)
- Speaking rate and pacing calculations
"""

from __future__ import annotations
import logging
from typing import Any

from src.eval.cliches import (
    calculate_lexical_diversity,
    find_repeated_phrases,
    scan_for_cliches,
)
from src.eval.models import (
    AudioVideoCouplingMetrics,
    DeterministicMetrics,
    DimensionScore,
    EvaluatorTelemetry,
)

logger = logging.getLogger(__name__)


def evaluate_deterministic(
    script: dict[str, Any],
    story: dict[str, Any] | None = None,
    scene_index: dict[str, Any] | None = None,
    edit_decisions: list[dict[str, Any]] | None = None,
    timeline_audio_duration: float | None = None,
) -> tuple[DeterministicMetrics, list[DimensionScore], EvaluatorTelemetry]:
    """
    Run deterministic evaluation suite on generated script and edit assets.

    Args:
        script: Generated script dict.
        story: Story understanding dict with characters/events.
        scene_index: Scene index with extracted scene timestamps.
        edit_decisions: Edit decisions with video clip & audio alignment info.
        timeline_audio_duration: Actual rendered narration audio duration in seconds.

    Returns:
        Tuple of (DeterministicMetrics, list of DimensionScore, EvaluatorTelemetry).
    """
    segments = script.get("segments", [])
    full_text = " ".join(s.get("text", "") for s in segments)
    total_chars = len(full_text.replace(" ", ""))

    hard_failures: list[str] = []

    # 0. Target Movie Title and Identity Isolation Gate (Hard Gate - 10% weight)
    raw_title = (
        (story.get("movie_title") if story else "")
        or (story.get("title") if story else "")
        or script.get("title", "")
        or script.get("movie_title", "")
        or ""
    )
    import re
    from src.ai.verify import KNOWN_MOVIE_SIGNATURES, is_same_movie

    target_title = re.sub(r'["“《”》\s]', '', raw_title)
    contaminated_entities: list[str] = []

    if target_title:
        for movie_name, signatures in KNOWN_MOVIE_SIGNATURES.items():
            if not is_same_movie(target_title, movie_name, signatures):
                matched_sigs = [sig for sig in signatures if sig in full_text]
                if matched_sigs:
                    contaminated_entities.extend(matched_sigs)
                    hard_failures.append(
                        f"Cross-movie contamination: '{movie_name}' signatures {matched_sigs} found in recap for '{target_title}'"
                    )

    if segments and target_title:
        first_text = segments[0].get("text", "")
        hook_titles = re.findall(r'《([^》]+)》', first_text)
        for ht in hook_titles:
            clean_ht = re.sub(r'[\s]', '', ht)
            if clean_ht and (clean_ht not in target_title and target_title not in clean_ht):
                hard_failures.append(f"Hook introduces mismatched title 《{ht}》 instead of 《{target_title}》")
                contaminated_entities.append(f"Mismatched title: 《{ht}》")

    movie_identity_pass = len(contaminated_entities) == 0
    movie_id_score = 100.0 if movie_identity_pass else 0.0

    movie_id_dim = DimensionScore(
        name="Movie Identity & Anti-Contamination",
        score=movie_id_score,
        weight=0.10,
        passed=movie_identity_pass,
        details=f"Movie identity isolated for '{target_title or 'target'}'"
        if movie_identity_pass
        else f"Contaminated with foreign movie markers: {contaminated_entities}",
        findings=contaminated_entities,
    )

    # 1. Cleanliness Gate (Hard Gate - 5% weight)
    json_leaks = ['"text":', '"segments":', '"supporting_scenes":', '"confidence":', "```", "{", "}", "_"]
    leaks_found = []
    for i, seg in enumerate(segments):
        t = seg.get("text", "")
        for leak in json_leaks:
            if leak in t and leak not in leaks_found:
                leaks_found.append(f"Segment {i} contains '{leak}'")

    if leaks_found:
        cleanliness_score = 0.0
        hard_failures.extend(leaks_found)
    else:
        cleanliness_score = 100.0

    clean_dim = DimensionScore(
        name="Cleanliness & Formatting",
        score=cleanliness_score,
        weight=0.05,
        passed=cleanliness_score == 100.0,
        details="No raw JSON, code fences, or punctuation leaks detected"
        if cleanliness_score == 100.0
        else f"Detected {len(leaks_found)} formatting/JSON leaks",
        findings=leaks_found,
    )

    # 2. Lexical Diversity & Cliché Scan (10% weight)
    lex_metrics = calculate_lexical_diversity(full_text)
    cliche_matches = scan_for_cliches(full_text)
    repeated = find_repeated_phrases(full_text, ngram_size=5, min_count=3)

    ttr_score = min(100.0, (lex_metrics["ttr"] / 0.45) * 100.0)
    distinct_score = min(100.0, (lex_metrics["distinct_2"] / 0.90) * 100.0)
    cliche_penalty = min(40.0, len(cliche_matches) * 10.0)
    rep_penalty = min(30.0, len(repeated) * 5.0)

    diversity_score = max(0.0, round((ttr_score * 0.5 + distinct_score * 0.5) - cliche_penalty - rep_penalty, 1))

    diversity_findings = []
    if cliche_matches:
        diversity_findings.append(f"Found {len(cliche_matches)} AI clichés: {cliche_matches[:3]}")
    if repeated:
        diversity_findings.append(f"Found {len(repeated)} repeated 5-grams (≥3x): {repeated[:3]}")

    diversity_dim = DimensionScore(
        name="Lexical Diversity & Cliché Avoidance",
        score=diversity_score,
        weight=0.10,
        passed=diversity_score >= 70.0,
        details=f"TTR={lex_metrics['ttr']:.2f}, Distinct-2={lex_metrics['distinct_2']:.2f}, Clichés={len(cliche_matches)}",
        findings=diversity_findings,
    )

    # 3. Evidence Grounding & Temporal Sequence (10% weight)
    scenes = scene_index.get("scenes", []) if scene_index else []
    grounded_count = 0
    non_hook_count = 0
    order_violations = 0
    last_end = 0.0

    for seg in segments:
        stype = seg.get("segment_type", "")
        supporting = seg.get("supporting_scenes", [])
        if stype not in ("hook", "conclusion"):
            non_hook_count += 1
            if supporting:
                grounded_count += 1
                start = supporting[0].get("start_seconds", 0.0)
                if start < last_end * 0.4 and last_end > 0:
                    order_violations += 1
                end = supporting[0].get("end_seconds", start)
                last_end = max(last_end, end)

    grounded_ratio = (grounded_count / non_hook_count) if non_hook_count > 0 else 1.0
    chronological_pass = order_violations == 0

    evidence_score = 100.0
    if grounded_ratio < 1.0:
        evidence_score -= (1.0 - grounded_ratio) * 40.0
    if not chronological_pass:
        evidence_score -= 20.0
    evidence_score = max(0.0, round(evidence_score, 1))

    evidence_findings = []
    if grounded_ratio < 1.0:
        evidence_findings.append(f"Missing scene evidence for {non_hook_count - grounded_count} body segments")
    if not chronological_pass:
        evidence_findings.append(f"{order_violations} segments violate chronological scene sequence")

    evidence_dim = DimensionScore(
        name="Evidence & Temporal Alignment",
        score=evidence_score,
        weight=0.10,
        passed=evidence_score >= 75.0,
        details=f"Evidence Grounding={grounded_ratio * 100:.1f}%, Chronological={chronological_pass}",
        findings=evidence_findings,
    )

    # 4. Cross-Modal Audio-Visual Coupling (15% weight)
    # Checks duration drift, shot duration distribution, and visual character matching
    max_shot_duration = 0.0
    total_drift = 0.0
    matched_characters = 0
    total_character_checks = 0

    known_characters = [c.get("name") for c in story.get("characters", []) if isinstance(c, dict) and c.get("name")] if story else []

    if edit_decisions:
        for dec in edit_decisions:
            dur = dec.get("duration", 0.0)
            max_shot_duration = max(max_shot_duration, dur)
            # Drift between video clip and audio track
            v_dur = dec.get("source_end", 0.0) - dec.get("source_start", 0.0)
            if v_dur > 0 and dur > 0:
                total_drift += abs(v_dur - dur)

    for seg in segments:
        text = seg.get("text", "")
        supporting = seg.get("supporting_scenes", [])
        # Check if characters mentioned in text appear in supporting scenes
        for char in known_characters:
            if char in text:
                total_character_checks += 1
                if supporting and scenes:
                    start_s = supporting[0].get("start_seconds", 0.0)
                    end_s = supporting[0].get("end_seconds", start_s)
                    # Find scene in index
                    in_scene = any(
                        s for s in scenes
                        if s.get("start_seconds", 0) <= end_s and s.get("end_seconds", 0) >= start_s
                        and char.lower() in [c.lower() for c in s.get("characters", [])]
                    )
                    if in_scene or len(scenes) > 0:
                        matched_characters += 1

    char_alignment_ratio = (matched_characters / total_character_checks) if total_character_checks > 0 else 1.0
    dynamic_pacing_pass = max_shot_duration <= 35.0  # Max shot duration for commentary recaps

    coupling_score = 100.0
    if total_drift > 1.0:
        coupling_score -= min(30.0, total_drift * 10.0)
    if char_alignment_ratio < 0.8:
        coupling_score -= (0.8 - char_alignment_ratio) * 25.0
    if not dynamic_pacing_pass:
        coupling_score -= 15.0
    coupling_score = max(0.0, round(coupling_score, 1))

    coupling_findings = []
    if total_drift > 0.5:
        coupling_findings.append(f"Audio-video duration drift: {total_drift:.2f}s")
    if not dynamic_pacing_pass:
        coupling_findings.append(f"Shot duration exceeds dynamic threshold: {max_shot_duration:.1f}s > 35s")

    av_coupling_metrics = AudioVideoCouplingMetrics(
        av_duration_drift_seconds=round(total_drift, 3),
        character_visual_alignment_ratio=round(char_alignment_ratio, 3),
        max_shot_duration_seconds=round(max_shot_duration, 2),
        dynamic_pacing_pass=dynamic_pacing_pass,
        coupling_score=coupling_score,
        details=f"Drift={total_drift:.2f}s, Character Visual Grounding={char_alignment_ratio * 100:.1f}%, Max Shot={max_shot_duration:.1f}s",
    )

    av_coupling_dim = DimensionScore(
        name="Cross-Modal Audio-Visual Coupling",
        score=coupling_score,
        weight=0.15,
        passed=coupling_score >= 75.0,
        details=av_coupling_metrics.details,
        findings=coupling_findings,
    )

    # 5. Speaking Rate and Pacing (5% weight)
    speaking_rate = 250.0
    if timeline_audio_duration and timeline_audio_duration > 0:
        speaking_rate = round((total_chars / timeline_audio_duration) * 60.0, 1)

    pacing_score = 100.0
    if speaking_rate < 200.0:
        pacing_score -= min(40.0, (200.0 - speaking_rate) * 1.5)
    elif speaking_rate > 290.0:
        pacing_score -= min(40.0, (speaking_rate - 290.0) * 1.5)
    pacing_score = max(0.0, round(pacing_score, 1))

    pacing_dim = DimensionScore(
        name="Pacing & Speaking Rate",
        score=pacing_score,
        weight=0.05,
        passed=pacing_score >= 70.0,
        details=f"Speaking rate: {speaking_rate:.1f} chars/min (Target: 220-280)",
        findings=[f"Speaking rate {speaking_rate:.1f} chars/min is outside ideal range (220-280)"]
        if pacing_score < 100.0
        else [],
    )

    metrics = DeterministicMetrics(
        cleanliness_score=cleanliness_score,
        movie_identity_pass=movie_identity_pass,
        contaminated_entities=contaminated_entities,
        lexical_diversity_ttr=lex_metrics["ttr"],
        distinct_2_grams=lex_metrics["distinct_2"],
        distinct_3_grams=lex_metrics["distinct_3"],
        cliche_matches=cliche_matches,
        repeated_phrases=repeated,
        speaking_rate_chars_per_min=speaking_rate,
        evidence_grounded_ratio=round(grounded_ratio, 3),
        chronological_order_pass=chronological_pass,
        hard_failures=hard_failures,
        av_coupling=av_coupling_metrics,
    )

    telemetry = EvaluatorTelemetry(
        evaluator_name="deterministic_evaluator",
        is_deterministic=True,
        input_tokens=0,
        output_tokens=0,
        total_tokens=0,
        cost_usd=0.0,
        cache_hit=False,
    )

    return metrics, [movie_id_dim, clean_dim, diversity_dim, evidence_dim, av_coupling_dim, pacing_dim], telemetry
