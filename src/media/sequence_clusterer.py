"""
Deterministic Sequence Clusterer for V3 Video-First Recap Engine (FR-7 / ADR 0006).

Groups granular shot-cuts (from PySceneDetect) into continuous, logical narrative sequences
based on subtitle dialogue continuity, character presence, and temporal boundaries.
Guarantees unbroken contiguous time ranges for every macro-scene.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class NarrativeSequence:
    """A continuous logical sequence / scene block in the source movie."""
    sequence_id: str
    sequence_index: int
    start_seconds: float
    end_seconds: float
    duration_seconds: float
    scene_ids: list[str] = field(default_factory=list)
    transcript_text: str = ""
    characters: list[str] = field(default_factory=list)
    keyframe_paths: list[str] = field(default_factory=list)
    dialogue_count: int = 0
    is_main_story: bool = True
    narrative_weight: float = 5.0
    event_type: str = "plot"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def cluster_scenes_into_sequences(
    scenes: list[dict[str, Any]],
    subtitles: Optional[list[dict[str, Any]]] = None,
    min_sequence_duration: float = 45.0,
    max_sequence_duration: float = 120.0,
    silence_gap_threshold: float = 4.0,
) -> list[NarrativeSequence]:
    """
    Cluster contiguous shot-cuts into narrative macro-scenes.

    Args:
        scenes: Ordered list of scene dicts from scene_index.
        subtitles: Optional list of subtitle entries with start/end/text.
        min_sequence_duration: Minimum duration (seconds) before a sequence can close.
        max_sequence_duration: Maximum duration (seconds) before forcing a sequence close.
        silence_gap_threshold: Dialogue silence gap (seconds) triggering a natural boundary.

    Returns:
        List of contiguous NarrativeSequence objects spanning the movie.
    """
    if not scenes:
        return []

    subs = subtitles or []
    sequences: list[NarrativeSequence] = []

    cur_scenes: list[dict[str, Any]] = []
    cur_start = float(scenes[0].get("start_seconds", 0.0))
    seq_counter = 0

    for i, sc in enumerate(scenes):
        sc_st = float(sc.get("start_seconds", 0.0))
        sc_et = float(sc.get("end_seconds", sc_st + 1.0))
        cur_scenes.append(sc)

        current_dur = sc_et - cur_start

        # Check if this scene boundary is a natural cut point:
        is_last_scene = (i == len(scenes) - 1)
        reached_max = current_dur >= max_sequence_duration

        # Check dialogue gap to next scene
        has_dialogue_break = False
        if current_dur >= min_sequence_duration and i < len(scenes) - 1:
            next_st = float(scenes[i + 1].get("start_seconds", sc_et))
            # Find if there are subtitles active between sc_et and next_st
            active_subs = [s for s in subs if sc_et - 1.0 <= s.get("start", 0.0) <= next_st + 1.0]
            if not active_subs:
                has_dialogue_break = True

        if is_last_scene or reached_max or (current_dur >= min_sequence_duration and has_dialogue_break):
            # Finalize current sequence
            seq_id = f"seq-{seq_counter:03d}"
            sc_ids = [s.get("scene_id", f"s-{j}") for j, s in enumerate(cur_scenes)]

            # Aggregate transcript and characters
            seq_transcripts = []
            seq_chars: set[str] = set()
            seq_keyframes: list[str] = []

            for s in cur_scenes:
                t = s.get("transcript_text", "").strip()
                if t:
                    seq_transcripts.append(t)
                for c in s.get("characters", []):
                    if c:
                        seq_chars.add(c)
                for kf in s.get("keyframe_paths", []):
                    if kf and str(kf) not in seq_keyframes:
                        seq_keyframes.append(str(kf))

            # If subtitles provided, extract subtitles strictly within [cur_start, sc_et]
            seq_subs = [s for s in subs if cur_start - 0.2 <= s.get("start", 0.0) <= sc_et + 0.2]
            if seq_subs and not seq_transcripts:
                seq_transcripts = [s.get("text", "") for s in seq_subs if s.get("text")]

            full_text = " ".join(seq_transcripts)
            diag_count = len(seq_subs) if seq_subs else len(seq_transcripts)

            seq = NarrativeSequence(
                sequence_id=seq_id,
                sequence_index=seq_counter,
                start_seconds=round(cur_start, 2),
                end_seconds=round(sc_et, 2),
                duration_seconds=round(sc_et - cur_start, 2),
                scene_ids=sc_ids,
                transcript_text=full_text,
                characters=sorted(list(seq_chars)),
                keyframe_paths=seq_keyframes,
                dialogue_count=diag_count,
            )
            sequences.append(seq)
            seq_counter += 1

            # Reset for next sequence
            cur_scenes = []
            if not is_last_scene:
                cur_start = float(scenes[i + 1].get("start_seconds", sc_et))

    logger.info(
        "Clustered %d scenes into %d narrative sequences (avg duration %.1fs)",
        len(scenes),
        len(sequences),
        sum(s.duration_seconds for s in sequences) / max(1, len(sequences)),
    )
    return sequences
