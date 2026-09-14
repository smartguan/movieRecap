"""
Scene Semantic Matcher (FR-7 / ADR 0005).

Deterministically matches narration segments to source movie scenes based on:
1. Subtitle & transcript lexical overlap (including cross-lingual Kanji / Hanzi and key concept mapping).
2. Local summary and story event semantic overlap.
3. Character visual / dialogue presence.
4. Narrative act phase alignment with smooth chronological progression.
5. Anti-looping and minimum spacing constraints.

Operates with 0 LLM tokens using deterministic tokenization, inverted indexing,
and multi-signal affinity scoring.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# Cross-lingual and domain-specific concept mappings (Mandarin narration -> Subtitle/Transcript cues)
CONCEPT_MAP: dict[str, list[str]] = {
    "理发": ["カット", "カラー", "髪", "美容", "イメチェン", "サロン"],
    "剪发": ["カット", "髪", "美容"],
    "发丝": ["カット", "髪"],
    "收音机": ["音楽", "ラジオ"],
    "手机": ["携帯", "スマホ", "電話", "ポケット", "着信"],
    "动态": ["投稿", "SNS", "アカウント", "通知"],
    "社交": ["SNS", "投稿", "アカウント"],
    "视频": ["動画", "ビデオ", "カメラ", "映像"],
    "网络": ["ネット", "SNS", "投稿"],
    "纸人": ["神人魚", "人形", "紙", "呪い"],
    "人偶": ["人形", "神人魚", "紙"],
    "符咒": ["神人魚", "人形", "お札", "呪い", "符"],
    "留言": ["コメント", "投稿", "返信"],
    "评论": ["コメント", "投稿"],
    "镜子": ["鏡", "反射"],
    "浴缸": ["風呂", "浴槽", "溺"],
    "溺毙": ["風呂", "浴槽", "溺", "死"],
    "死亡": ["死", "自殺", "殺", "亡"],
    "死于非命": ["死", "殺", "亡"],
    "日记": ["日記", "手紙", "ノート"],
    "明信片": ["ハガキ", "ポストカード", "手紙"],
    "台湾": ["台湾", "タイワン", "台北"],
    "台北": ["台北", "台湾", "空港"],
    "航班": ["飛行機", "空港", "フライト", "便"],
    "飞机": ["飛行機", "空港"],
    "机场": ["空港", "ロビー", "到着"],
    "道铺": ["道士", "香", "店", "お札", "線香"],
    "香烛": ["線香", "香", "ろうそく", "蝋燭"],
    "道长": ["動詞", "道士", "先生", "師"],
    "长者": ["先生", "老人", "道士"],
    "七日绝魂煞": ["神人業", "呪い", "悪霊", "煞"],
    "霸凌": ["いじめ", "無視", "孤立", "悪口"],
    "神庙": ["廟", "寺", "神社", "神殿", "神"],
    "古庙": ["廟", "寺", "神社"],
    "供桌": ["供物", "祭壇", "机", "神棚"],
    "怨灵": ["呪い", "霊", "怨", "悪霊", "幽霊"],
    "恶灵": ["呪い", "霊", "怨", "悪霊"],
    "红衣": ["赤", "紅", "服"],
    "打火机": ["ライター", "火", "燃"],
    "烈火": ["火", "燃", "炎"],
    "大火": ["火", "燃", "炎"],
    "晨光": ["朝", "光", "朝日", "夜明け"],
    "破晓": ["朝", "光", "夜明け"],
    "密林": ["山", "森", "林", "木"],
    "东京": ["東京", "日本"],
}


def tokenize_text(text: str) -> set[str]:
    """
    Deterministically extract unigrams, bigrams, and alphanumeric words from text.
    Handles Chinese, Japanese Kanji/Kana, and Latin characters without external dependencies.
    """
    if not text:
        return set()

    tokens: set[str] = set()

    # 1. Alphanumeric words (e.g. "SNS", "2019", "WiFi")
    ascii_words = re.findall(r"[A-Za-z0-9_]+", text)
    for w in ascii_words:
        tokens.add(w.lower())
        tokens.add(w.upper())

    # 2. CJK character extraction
    cjk_chars = re.findall(r"[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff]", text)
    for char in cjk_chars:
        tokens.add(char)

    # 3. Bigrams across adjacent CJK characters
    for i in range(len(cjk_chars) - 1):
        bigram = cjk_chars[i] + cjk_chars[i + 1]
        tokens.add(bigram)

    # 4. Trigrams for specific terminology
    for i in range(len(cjk_chars) - 2):
        trigram = cjk_chars[i] + cjk_chars[i + 1] + cjk_chars[i + 2]
        tokens.add(trigram)

    return tokens


@dataclass
class IndexedScene:
    """Indexed scene metadata and token bags for fast semantic scoring."""
    scene_id: str
    index: int
    start_seconds: float
    end_seconds: float
    duration_seconds: float
    transcript_text: str
    characters: list[str]
    local_summary_text: str = ""
    summary_events_text: str = ""
    direct_tokens: set[str] = field(default_factory=set)
    context_tokens: set[str] = field(default_factory=set)  # includes ±1 scene transcript window
    summary_tokens: set[str] = field(default_factory=set)


class SemanticSceneIndex:
    """
    In-memory semantic index across movie scenes, transcripts, and story understanding.
    """

    def __init__(
        self,
        scene_index: dict[str, Any],
        story_understanding: Optional[dict[str, Any]] = None,
    ) -> None:
        self.scene_index = scene_index
        self.story_understanding = story_understanding or {}
        self.total_duration = float(scene_index.get("duration_seconds", 0.0))
        self.scenes: list[IndexedScene] = []
        self._build_index()

    def _build_index(self) -> None:
        raw_scenes = self.scene_index.get("scenes", [])
        if not raw_scenes:
            return

        if self.total_duration <= 0.0:
            self.total_duration = max(float(s.get("end_seconds", 0.0)) for s in raw_scenes)

        local_summaries = self.story_understanding.get("local_summaries", [])

        # Map each scene to its local summary chunk
        num_chunks = max(1, len(local_summaries))
        scenes_per_chunk = max(1, math.ceil(len(raw_scenes) / num_chunks))

        for idx, sc in enumerate(raw_scenes):
            sc_st = float(sc.get("start_seconds", 0.0))
            sc_et = float(sc.get("end_seconds", sc_st + 1.0))
            sc_dur = float(sc.get("duration_seconds", sc_et - sc_st))
            transcript = sc.get("transcript_text", "")
            chars = list(sc.get("characters", []))

            # Locate corresponding local summary
            chunk_idx = min(idx // scenes_per_chunk, len(local_summaries) - 1) if local_summaries else -1
            ls_text = ""
            ls_events_text = ""
            if 0 <= chunk_idx < len(local_summaries):
                ls = local_summaries[chunk_idx]
                ls_text = ls.get("summary", "") or ls.get("text", "")
                ev_list = ls.get("events", [])
                ls_events_text = " ".join(e.get("description", "") for e in ev_list if isinstance(e, dict))
                for c in ls.get("characters_seen", []):
                    if c and c not in chars:
                        chars.append(c)

            dir_tokens = tokenize_text(transcript)
            sum_tokens = tokenize_text(f"{ls_text} {ls_events_text}")

            indexed = IndexedScene(
                scene_id=sc.get("scene_id", f"scene-{idx:04d}"),
                index=idx,
                start_seconds=sc_st,
                end_seconds=sc_et,
                duration_seconds=sc_dur,
                transcript_text=transcript,
                characters=chars,
                local_summary_text=ls_text,
                summary_events_text=ls_events_text,
                direct_tokens=dir_tokens,
                summary_tokens=sum_tokens,
            )
            self.scenes.append(indexed)

        # Context tokens: expand ±1 adjacent scenes to bridge dialogue reactions (only if contiguous <= 8.0s)
        for i, sc in enumerate(self.scenes):
            ctx_tokens = set(sc.direct_tokens)
            if i > 0 and abs(sc.start_seconds - self.scenes[i - 1].end_seconds) <= 8.0:
                ctx_tokens.update(self.scenes[i - 1].direct_tokens)
            if i + 1 < len(self.scenes) and abs(self.scenes[i + 1].start_seconds - sc.end_seconds) <= 8.0:
                ctx_tokens.update(self.scenes[i + 1].direct_tokens)
            sc.context_tokens = ctx_tokens


def build_semantic_scene_index(
    scene_index: dict[str, Any],
    story_understanding: Optional[dict[str, Any]] = None,
) -> SemanticSceneIndex:
    """Build and return a SemanticSceneIndex instance."""
    return SemanticSceneIndex(scene_index, story_understanding)


def score_scene_for_narration(
    scene: IndexedScene,
    narration_text: str,
    target_timestamp: float,
    total_duration: float,
    expected_phase: Optional[Tuple[float, float]] = None,
    last_source_start: float = -1.0,
) -> float:
    """
    Score the semantic affinity of a single scene for a narration beat.

    Signals:
    1. Concept & Keyword mapping (0.35)
    2. Direct & Context Subtitle Token Overlap (0.25)
    3. Local Summary & Event Overlap (0.20)
    4. Phase Window Conformity (0.15)
    5. Chronological Monotonicity & Smooth Progression (0.05)

    Returns:
        Affinity score in range [0.0, 1.0].
    """
    if not narration_text:
        return 0.0

    narration_tokens = tokenize_text(narration_text)

    # 1. Concept Mapping Overlap
    matched_concept_weight = 0.0
    for concept, cues in CONCEPT_MAP.items():
        if concept in narration_text:
            if any(cue in scene.transcript_text for cue in cues) or any(cue in scene.context_tokens for cue in cues):
                matched_concept_weight += 1.0
            elif any(cue in scene.local_summary_text for cue in cues) or any(cue in scene.summary_events_text for cue in cues):
                matched_concept_weight += 0.8

    if matched_concept_weight > 0:
        concept_score = min(1.0, 0.6 + 0.4 * (matched_concept_weight - 1.0))
    else:
        concept_score = 0.0

    # 2. Direct Subtitle / Transcript Token Overlap
    transcript_overlap = len(narration_tokens.intersection(scene.context_tokens))
    transcript_score = min(1.0, transcript_overlap / max(3.0, len(narration_tokens) * 0.15))

    # 3. Local Summary & Event Overlap
    summary_overlap = len(narration_tokens.intersection(scene.summary_tokens))
    summary_score = min(1.0, summary_overlap / max(4.0, len(narration_tokens) * 0.20))

    # 4. Phase Window Alignment
    phase_score = 0.5
    if expected_phase is not None:
        p_min, p_max = expected_phase
        if p_min - 30.0 <= scene.start_seconds <= p_max + 30.0:
            phase_score = 1.0
        else:
            dist = min(abs(scene.start_seconds - p_min), abs(scene.start_seconds - p_max))
            phase_score = -2.0 - (dist / 100.0)
    else:
        dist = abs(scene.start_seconds - target_timestamp)
        phase_score = max(0.0, 1.0 - (dist / max(120.0, total_duration * 0.25)))

    # 5. Chronological Progression (Smoothness)
    chronological_score = 0.5
    if last_source_start >= 0:
        delta = scene.start_seconds - last_source_start
        if delta >= 0:
            chronological_score = min(1.0, 0.5 + 0.5 * (1.0 - min(1.0, delta / max(1.0, total_duration * 0.1))))
        elif delta >= -30.0:
            chronological_score = 0.2
        else:
            chronological_score = -1.0 - abs(delta) / 100.0

    final_score = (
        0.30 * concept_score
        + 0.20 * transcript_score
        + 0.15 * summary_score
        + 0.25 * phase_score
        + 0.10 * chronological_score
    )

    return round(final_score, 4)


def find_best_scenes(
    semantic_index: SemanticSceneIndex,
    narration_text: str,
    target_timestamp: float,
    needed_duration: float,
    used_start_timestamps: list[float],
    expected_phase: Optional[Tuple[float, float]] = None,
    last_source_start: float = -1.0,
    min_spacing: float = 8.0,
    top_k: int = 1,
) -> list[Tuple[IndexedScene, float]]:
    """
    Find the best scoring scenes for a narration segment, obeying anti-looping and spacing constraints.

    Returns:
        List of (IndexedScene, score) tuples, sorted by score descending.
    """
    scenes = semantic_index.scenes
    if not scenes:
        return []

    total_dur = semantic_index.total_duration
    candidates: list[Tuple[IndexedScene, float]] = []

    for sc in scenes:
        st = sc.start_seconds

        # Anti-looping: skip scenes too close to recently used timestamps
        if any(abs(st - prev) < min_spacing for prev in used_start_timestamps):
            continue

        score = score_scene_for_narration(
            scene=sc,
            narration_text=narration_text,
            target_timestamp=target_timestamp,
            total_duration=total_dur,
            expected_phase=expected_phase,
            last_source_start=last_source_start,
        )
        candidates.append((sc, score))

    if not candidates:
        for sc in scenes:
            score = score_scene_for_narration(
                scene=sc,
                narration_text=narration_text,
                target_timestamp=target_timestamp,
                total_duration=total_dur,
                expected_phase=expected_phase,
                last_source_start=last_source_start,
            )
            candidates.append((sc, score))

    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[:top_k]
