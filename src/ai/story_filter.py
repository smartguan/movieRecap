"""
Main Story Filter & Key Moment Selector for V3 Engine (FR-7 / ADR 0006).

Identifies the primary narrative spine vs. sideline filler/subplots,
scores dramatic intensity, and selects a chronologically ordered sequence set
that fits the target duration budget (1/5 of source runtime).
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from src.ai.gateway import LLMGateway
from src.media.scene_matcher import CONCEPT_MAP
from src.media.sequence_clusterer import NarrativeSequence

logger = logging.getLogger(__name__)


PLOT_TURN_KEYWORDS = [
    "死", "呪", "写真", "神人", "自殺", "道士", "下咒", "台湾", "連絡", "犯人", "アカウント", "殺", "真相",
    "食べる", "料理", "ご飯", "カット", "髪", "投稿", "SNS", "変な", "病院", "動画",
    "die", "kill", "curse", "truth", "murder", "photo", "ghost", "witch", "evil", "suicide"
]


def score_sequence_importance(
    sequence: NarrativeSequence,
    story_understanding: dict[str, Any],
    total_movie_duration: float,
) -> tuple[bool, float, str]:
    """
    Deterministically score sequence narrative importance based on plot-turn keywords and dialogue density.

    Returns:
        (is_main_story, narrative_weight [1.0 - 10.0], event_type)
    """
    text = sequence.transcript_text or ""
    st = sequence.start_seconds
    dur = sequence.duration_seconds

    # Check plot-turn keyword hits
    matched_plot_kws = [kw for kw in PLOT_TURN_KEYWORDS if kw in text]
    plot_hits = len(matched_plot_kws)

    # Character presence
    main_chars = story_understanding.get("characters", [])
    main_char_names = [c.get("name", "") for c in main_chars if isinstance(c, dict)]
    has_main_char = any(name in text for name in main_char_names if name) or bool(sequence.characters)

    # Determine is_main_story
    is_silent = len(text.strip()) == 0 and sequence.dialogue_count == 0
    if is_silent and dur < 60.0 and plot_hits == 0:
        return False, 1.0, "filler"

    # Base weight calculation driven by conversational and dramatic richness
    weight = 5.0

    # Domestic setting anchor bonus for setup phase (grounding ordinary life before terror)
    if any(w_c in text for w_c in ["食べる", "料理", "ご飯", "カット", "髪"]) and st < 500.0:
        weight += 2.5

    if plot_hits > 0:
        weight += min(4.0, plot_hits * 0.8)

    # Dialogue richness (conversation drives the plot in narrative films)
    if sequence.dialogue_count >= 15 or len(text) >= 200:
        weight += 3.0
    elif sequence.dialogue_count >= 8 or len(text) >= 100:
        weight += 2.0
    elif sequence.dialogue_count >= 3:
        weight += 1.0

    if has_main_char:
        weight += 0.5
    if is_silent:
        weight -= 2.0

    weight = max(1.0, min(10.0, round(weight, 1)))
    is_main = weight >= 4.5

    # Determine event type based on movie phase
    ratio = st / max(1.0, total_movie_duration)
    if ratio < 0.18:
        ev_type = "setup"
    elif ratio < 0.45:
        ev_type = "inciting_incident"
    elif ratio < 0.75:
        ev_type = "investigation"
    elif ratio < 0.90:
        ev_type = "climax"
    else:
        ev_type = "resolution"

    return is_main, weight, ev_type


def filter_main_story_sequences(
    sequences: list[NarrativeSequence],
    story_understanding: dict[str, Any],
    target_duration_sec: float,
    config: Optional[dict[str, Any]] = None,
) -> list[NarrativeSequence]:
    """
    Filter and select the optimal set of continuous sequences matching the target duration.

    Guarantees chronological narrative flow, dialogue-driven key plot turning points,
    and prevents narrow time clustering.
    """
    if not sequences:
        return []

    tot_dur = max(s.end_seconds for s in sequences)

    # 1. Score each sequence
    scored_seqs: list[NarrativeSequence] = []
    for s in sequences:
        is_main, weight, ev_type = score_sequence_importance(s, story_understanding, tot_dur)
        s.is_main_story = is_main
        s.narrative_weight = weight
        s.event_type = ev_type
        scored_seqs.append(s)

    # 2. Group into 5 chronological phases to guarantee narrative completeness
    phases: dict[str, list[NarrativeSequence]] = {
        "setup": [],
        "inciting_incident": [],
        "investigation": [],
        "climax": [],
        "resolution": [],
    }
    for s in scored_seqs:
        phases.get(s.event_type, phases["investigation"]).append(s)

    # 3. Budget allocation across phases
    phase_weights = {
        "setup": 0.22,
        "inciting_incident": 0.26,
        "investigation": 0.28,
        "climax": 0.16,
        "resolution": 0.08,
    }

    selected_seqs: list[NarrativeSequence] = []
    current_total_duration = 0.0

    # Ensure opening sequence (e.g. hair salon) is included if present
    opening_cands = [s for s in phases["setup"] if s.start_seconds < 300.0 and s.narrative_weight >= 7.0]
    opening_seq = sorted(opening_cands, key=lambda s: s.narrative_weight, reverse=True)[0] if opening_cands else None

    for phase_name, p_weight in phase_weights.items():
        phase_candidates = phases[phase_name]
        if not phase_candidates:
            continue

        target_phase_budget = target_duration_sec * p_weight
        # Sort candidates in this phase by narrative weight descending
        sorted_candidates = sorted(phase_candidates, key=lambda s: s.narrative_weight, reverse=True)

        phase_spent = 0.0
        phase_picked: list[NarrativeSequence] = []

        # If opening sequence belongs to setup, pick it first
        if phase_name == "setup" and opening_seq and opening_seq in sorted_candidates:
            phase_picked.append(opening_seq)
            phase_spent += opening_seq.duration_seconds

        for cand in sorted_candidates:
            if cand in phase_picked:
                continue
            if not cand.is_main_story and len(phase_picked) >= 1:
                continue
            # Enforce temporal dispersion: avoid clustering two sequences within 30s
            if any(abs(cand.start_seconds - p.start_seconds) < 30.0 for p in phase_picked):
                continue

            if phase_spent + cand.duration_seconds <= target_phase_budget * 1.30 or not phase_picked:
                phase_picked.append(cand)
                phase_spent += cand.duration_seconds
                if phase_spent >= target_phase_budget:
                    break

        selected_seqs.extend(phase_picked)
        current_total_duration += phase_spent

    # 4. Sort selected sequences strictly chronologically
    selected_seqs.sort(key=lambda s: s.start_seconds)

    logger.info(
        "V3 Story Filter: selected %d key sequences totaling %.1fs (target: %.1fs)",
        len(selected_seqs),
        sum(s.duration_seconds for s in selected_seqs),
        target_duration_sec,
    )
    return selected_seqs
