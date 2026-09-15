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
    scan_for_meta_commentary,
)
from src.eval.models import (
    AudioVideoCouplingMetrics,
    DeterministicMetrics,
    DimensionScore,
    EvaluatorTelemetry,
    StoryContinuityMetrics,
)

logger = logging.getLogger(__name__)


# Generic Activity / Setting Incompatibility Rules across Movie Recap Domains
ACTIVITY_DOMAINS: dict[str, dict[str, Any]] = {
    "dining_cooking": {
        "text_cues": ["做饭", "下厨", "烹饪", "晚餐", "晚饭", "午餐", "便当", "吃饭", "家常饭", "餐桌", "美食", "煮", "炒", "煎", "就餐", "享用午餐", "享用晚餐", "家常便饭"],
        "scene_cues": ["食べる", "食べた", "料理", "ご飯", "めし", "できた", "いただきます", "サイコーン", "おいしい", "キッチン", "kitchen", "cook", "dining", "dinner", "lunch", "meal"],
        "conflicts_with": ["salon_haircut"],
    },
    "salon_haircut": {
        "text_cues": ["理发", "剪发", "修剪发丝", "发丝", "理发师", "发型", "洗头", "吹风", "发廊", "发店"],
        "scene_cues": ["カット", "髪", "美容", "シャンプー", "パーマ", "イメチェン", "hair", "salon", "barber"],
        "conflicts_with": ["dining_cooking", "combat_horror", "temple_ritual"],
    },
    "phone_digital": {
        "text_cues": ["手机", "社交账号", "网络动态", "私密照片", "短信", "震动提示", "倒计时", "屏幕弹窗", "发帖", "推特", "照片"],
        "scene_cues": ["投稿", "SNS", "写真", "スマホ", "携帯", "アカウント", "ネット", "カメラ", "phone", "post", "screen"],
        "conflicts_with": [],
    },
    "travel_transit": {
        "text_cues": ["机场", "飞机", "航班", "降落", "起飞", "航站楼", "跨海", "桃园", "骑楼", "巷道", "街头", "驱车", "车窗"],
        "scene_cues": ["空港", "飛行機", "フライト", "台湾", "台北", "車", "道路", "airport", "flight", "plane", "travel", "driving"],
        "conflicts_with": [],
    },
    "temple_ritual": {
        "text_cues": ["神庙", "庙宇", "神龛", "香烛", "道铺", "符咒", "纸人", "母偶", "供桌", "封印", "阴煞"],
        "scene_cues": ["廟", "寺", "神社", "神棚", "お札", "符", "母偶", "人形", "線香", "お参り", "temple", "shrine", "altar", "ritual"],
        "conflicts_with": ["salon_haircut"],
    },
    "combat_horror": {
        "text_cues": ["搏杀", "打斗", "厉鬼", "怨灵", "火海", "焚毁", "点燃", "打火机", "尖叫", "惨死", "自尽", "血泊", "凶煞", "大殿崩塌"],
        "scene_cues": ["殺", "死", "火", "燃", "ライター", "幽霊", "怨霊", "怖い", "怪奇", "fight", "ghost", "flame", "corpse"],
        "conflicts_with": ["salon_haircut", "dining_cooking", "hospital_medical"],
    },
    "hospital_medical": {
        "text_cues": ["医院", "医生", "生理指标", "化验", "专科医院", "病理原因", "病因", "检查", "诊断", "医学"],
        "scene_cues": ["検査", "病院", "原因", "医者", "病気", "推薦", "hospital", "doctor", "medical"],
        "conflicts_with": ["salon_haircut", "dining_cooking", "combat_horror"],
    },
    "stalking_apparition": {
        "text_cues": ["如影随形", "床边", "独自沐浴", "阴冷的视线", "恶灵缠身", "诡异女子", "诡异女人", "梦魇"],
        "scene_cues": ["変な女", "突きまとって", "横にいる", "お風呂", "後ろに", "寝てても", "目が覚め", "ghost", "stalk"],
        "conflicts_with": ["dining_cooking", "salon_haircut"],
    },
    "suicide_tragedy": {
        "text_cues": ["自尽", "惨死在血泊", "自杀", "割腕", "跳楼", "死状", "自戕"],
        "scene_cues": ["自殺", "殺した", "死んじゃって", "suicide", "corpse"],
        "conflicts_with": ["dining_cooking", "salon_haircut"],
    },
}


def check_av_semantic_alignment(
    segments: list[dict[str, Any]],
    edit_decisions: list[dict[str, Any]],
    total_source_duration: float,
    movie_title: str = "",
    scene_index: Optional[dict[str, Any]] = None,
    story_understanding: Optional[dict[str, Any]] = None,
) -> tuple[float, int, list[str]]:
    """
    Audit whether visual clip timestamps align with narration semantic phases,
    activity domains, and scene content (ADR 0004, ADR 0005, ADR 0007).

    Returns:
        (alignment_score, mismatch_count, list of findings)
    """
    if not edit_decisions or not segments:
        return 100.0, 0, []

    from src.media.scene_matcher import (
        CONCEPT_MAP,
        build_semantic_scene_index,
        score_scene_for_narration,
    )

    findings: list[str] = []
    mismatches = 0
    total_checks = 0

    semantic_idx = None
    if scene_index and scene_index.get("scenes"):
        semantic_idx = build_semantic_scene_index(scene_index, story_understanding)

    for i, dec in enumerate(edit_decisions):
        text = dec.get("text", "")
        src_st = dec.get("source_start", 0.0)

        # Locate the scene at source_start
        matching_sc = None
        if semantic_idx and semantic_idx.scenes:
            matching_sc = next(
                (s for s in semantic_idx.scenes if s.start_seconds <= src_st <= s.end_seconds),
                None,
            )
            if matching_sc is None:
                matching_sc = min(semantic_idx.scenes, key=lambda s: abs(s.start_seconds - src_st))

        scene_text = matching_sc.transcript_text if matching_sc else ""
        scene_context_str = scene_text
        if matching_sc and hasattr(matching_sc, "context_tokens") and matching_sc.context_tokens:
            scene_context_str = " ".join(matching_sc.context_tokens) + " " + scene_text

        # 1. Generic Cross-Modal Activity and Setting Conflict Audit
        text_domains = {
            dom for dom, spec in ACTIVITY_DOMAINS.items()
            if any(cue in text for cue in spec["text_cues"])
        }
        scene_domains = {
            dom for dom, spec in ACTIVITY_DOMAINS.items()
            if any(cue in scene_context_str for cue in spec["scene_cues"])
        }

        for t_dom in text_domains:
            conflicting_doms = ACTIVITY_DOMAINS[t_dom]["conflicts_with"]
            for s_dom in scene_domains:
                if s_dom in conflicting_doms and s_dom not in text_domains:
                    total_checks += 1
                    mismatches += 1
                    findings.append(
                        f"AV Activity Conflict: Clip {i} ('{text[:24]}...') asserts '{t_dom}' activity, "
                        f"but video footage at {src_st:.1f}s contains '{s_dom}' dialogue/actions"
                    )

        # 2. Direct Scene Content Semantic Audit (when scene dialogue is present)
        if matching_sc and scene_text.strip():
            total_checks += 1
            c_score = score_scene_for_narration(
                scene=matching_sc,
                narration_text=text,
                target_timestamp=src_st,
                total_duration=total_source_duration,
            )
            if len(scene_text) >= 12 and c_score < 0.15:
                if not any(f"Clip {i}" in f for f in findings):
                    mismatches += 1
                    findings.append(
                        f"AV Scene Content Mismatch: Clip {i} ('{text[:24]}...') at {src_st:.1f}s has near-zero semantic correlation ({c_score:.2f}) with dialogue in scene {matching_sc.scene_id}"
                    )

        # 3. Macro Chronological Phase Audit across movie runtime
        if total_source_duration > 300.0:
            rel_pos = src_st / total_source_duration
            # A: Opening exposition scene placed at the end of movie
            if ("salon_haircut" in text_domains or any(k in text for k in ["故事从", "拉开帷幕", "起初", "修剪发丝"])) and rel_pos >= 0.70:
                total_checks += 1
                mismatches += 1
                findings.append(
                    f"AV Chronological Disconnect: Early exposition ('{text[:24]}...') placed at {src_st:.1f}s (>{total_source_duration * 0.70:.1f}s, {rel_pos*100:.1f}% into runtime)"
                )
            # B: Final climax / resolution placed in first 25% of movie
            elif ("combat_horror" in text_domains or any(k in text for k in ["决战", "怨灵决战", "防风打火机", "点燃母偶", "母偶化为灰烬", "晨光穿透"])) and rel_pos <= 0.25:
                total_checks += 1
                mismatches += 1
                findings.append(
                    f"AV Chronological Disconnect: Climax resolution ('{text[:24]}...') placed at {src_st:.1f}s (<{total_source_duration * 0.25:.1f}s, {rel_pos*100:.1f}% into runtime)"
                )
            # C: Distant overseas travel placed in initial 5% of movie
            elif any(k in text for k in ["跨海抵达", "降落在台北", "老旧街巷香烛道铺"]) and rel_pos <= 0.05:
                total_checks += 1
                mismatches += 1
                findings.append(
                    f"AV Chronological Disconnect: Overseas investigation ('{text[:24]}...') placed at {src_st:.1f}s (<{total_source_duration * 0.05:.1f}s, {rel_pos*100:.1f}% into runtime)"
                )

        # 4. First and last clip phase sanity (when full recap >= 3 clips)
        if len(edit_decisions) >= 3:
            if i == 0 and total_source_duration > 180.0 and src_st > total_source_duration * 0.25:
                total_checks += 1
                mismatches += 1
                findings.append(
                    f"AV Hook Disconnect: Hook clip at {src_st:.1f}s is outside introductory phase (>{total_source_duration * 0.25:.1f}s)"
                )
            elif i == len(edit_decisions) - 1 and total_source_duration > 180.0 and src_st < total_source_duration * 0.70:
                total_checks += 1
                mismatches += 1
                findings.append(
                    f"AV Conclusion Disconnect: Conclusion clip at {src_st:.1f}s is outside resolution phase (<{total_source_duration * 0.70:.1f}s)"
                )

    if total_checks == 0:
        return 100.0, 0, []

    alignment_ratio = max(0.0, (total_checks - mismatches) / total_checks)
    score = round(alignment_ratio * 100.0, 1)
    return score, mismatches, findings


TRANSITION_MARKERS: list[str] = [
    # Temporal & sequence progression
    "随后", "与此同时", "转眼间", "不久后", "几天后", "第二天", "次日", "几天过去",
    "数日后", "几年后", "多年后", "紧接着", "未等", "正在这时", "就在此时", "随着",
    "当夜", "午夜", "翌日", "数月后", "后来", "在此期间", "夜幕降临", "画面一转",
    "此时", "另一边", "几经周折", "顺着线索", "时间一晃", "随着时间",
    # Causal & motivational transitions
    "为了", "在得知", "在发现", "在经历", "见此情形", "于是", "因而", "为此", "决定",
    "为了弄清", "为了查明", "为了斩断", "为了救", "为了找到", "在惨烈", "在噩梦",
    # Spatial & journey movements
    "来到", "赶往", "孤身前往", "驱车", "跨海", "深入", "穿行于", "回到", "进入", "踏入",
    "走出", "启程", "奔赴", "飞往", "前往", "跨越", "降落", "飞赴", "穿行", "飞机",
    # Dramatic phase and narrative bridges
    "随着调查深入", "惨烈搏杀过后", "未等众人喘息", "尘埃落定", "生死一线之际",
    "随着真相浮出水面", "在真相大白之后", "回顾", "面对", "日以继夜", "惨剧发生之后", "悲剧发生后",
]


def calculate_storyteller_continuity(
    segments: list[dict[str, Any]],
    story: dict[str, Any] | None = None,
    edit_decisions: list[dict[str, Any]] | None = None,
    total_source_duration: float = 300.0,
) -> tuple[StoryContinuityMetrics, DimensionScore]:
    """
    Deterministically analyze the narrative and temporal continuity of the storyteller.

    Evaluates:
    1. Temporal monotonicity (absence of confusing backward timeline regressions)
    2. Discourse transition coherence on temporal/spatial jumps (gap >= 45s)
    3. Character and narrative entity continuity across adjacent beats
    4. Visual narrative spine contiguity (macro-scene stability vs. micro-fragmentation)

    Returns:
        (StoryContinuityMetrics, DimensionScore)
    """
    if not segments:
        m = StoryContinuityMetrics()
        d = DimensionScore(
            name="Storyteller Narrative Continuity",
            score=100.0,
            weight=0.12,
            passed=True,
            details="No segments to evaluate",
        )
        return m, d

    discontinuity_events: list[str] = []

    # 1. Resolve timestamp bounds for each segment
    timed_segments: list[tuple[int, dict[str, Any], float, float]] = []
    for i, seg in enumerate(segments):
        st = None
        et = None
        supporting = seg.get("supporting_scenes", [])
        if supporting and isinstance(supporting, list):
            st = float(supporting[0].get("start_seconds", 0.0))
            et = float(supporting[0].get("end_seconds", st))
        elif edit_decisions:
            seg_id = seg.get("segment_id", "")
            dec = next((d for d in edit_decisions if d.get("segment_id") == seg_id), None)
            if dec is None and i < len(edit_decisions):
                dec = edit_decisions[i]
            if dec:
                st = float(dec.get("source_start", 0.0))
                et = float(dec.get("source_end", st))

        if st is not None and et is not None:
            timed_segments.append((i, seg, st, et))

    # 2. Evaluate Temporal Monotonicity & Backward Jump Detection
    temporal_score = 100.0
    backward_jumps = 0
    for idx in range(1, len(timed_segments)):
        prev_i, _, prev_st, _ = timed_segments[idx - 1]
        curr_i, _, curr_st, _ = timed_segments[idx]

        delta = curr_st - prev_st
        if delta < -2.0:
            backward_gap = abs(delta)
            if backward_gap > 15.0:
                backward_jumps += 1
                temporal_score -= 25.0
                discontinuity_events.append(
                    f"Major backward timeline regression: Segment {curr_i} ({curr_st:.1f}s) jumps backward by {backward_gap:.1f}s behind Segment {prev_i} ({prev_st:.1f}s)"
                )
            else:
                temporal_score -= 5.0

    temporal_monotonicity_score = max(0.0, min(100.0, round(temporal_score, 1)))

    # 3. Evaluate Transition Coherence on Scene Jumps (gap >= 45s)
    scene_jumps = 0
    unbridged_jumps = 0
    for idx in range(1, len(timed_segments)):
        prev_i, _, _, prev_et = timed_segments[idx - 1]
        curr_i, curr_seg, curr_st, _ = timed_segments[idx]

        forward_gap = curr_st - prev_et
        if forward_gap >= 45.0:
            scene_jumps += 1
            curr_text = curr_seg.get("text", "")
            has_transition = (
                any(marker in curr_text[:50] for marker in TRANSITION_MARKERS)
                or any(marker in curr_text for marker in ["随着", "为了", "来到", "赶往", "跨海", "深入", "随后", "与此同时", "决定", "后来", "回顾"])
            )
            if not has_transition:
                unbridged_jumps += 1
                discontinuity_events.append(
                    f"Unbridged narrative jump: Segment {curr_i} leaps forward by {forward_gap:.1f}s without connective transition phrasing"
                )

    if scene_jumps == 0:
        transition_coherence_ratio = 1.0
    else:
        transition_coherence_ratio = round(max(0.0, (scene_jumps - unbridged_jumps) / scene_jumps), 3)

    # 4. Character & Entity Narrative Threading
    known_chars: list[str] = []
    if story and isinstance(story, dict):
        raw_chars = story.get("characters", [])
        if isinstance(raw_chars, list):
            for c in raw_chars:
                if isinstance(c, dict) and c.get("name"):
                    known_chars.append(c.get("name"))
                elif isinstance(c, str) and c:
                    known_chars.append(c)

    unthreaded_pairs = 0
    total_pairs = max(0, len(segments) - 1)
    if total_pairs > 0 and known_chars:
        for idx in range(1, len(segments)):
            prev_text = segments[idx - 1].get("text", "")
            curr_text = segments[idx].get("text", "")

            prev_has = [c for c in known_chars if c in prev_text]
            curr_has = [c for c in known_chars if c in curr_text]

            # If both mention characters but have zero overlap and no scene shift transition
            if prev_has and curr_has and not (set(prev_has) & set(curr_has)):
                has_shift_marker = any(m in curr_text[:40] for m in ["另一边", "与此同时", "此时", "来到", "随着", "而"])
                if not has_shift_marker:
                    unthreaded_pairs += 1
                    discontinuity_events.append(
                        f"Entity focus shift: Segment {idx} switches from {prev_has} to {curr_has} without transition bridge"
                    )

        character_entity_thread_ratio = round(max(0.0, (total_pairs - unthreaded_pairs) / total_pairs), 3)
    else:
        character_entity_thread_ratio = 1.0

    # 5. Visual Narrative Spine Contiguity
    durations: list[float] = []
    if edit_decisions:
        for d in edit_decisions:
            dur = float(d.get("duration", 0.0))
            if dur <= 0.0:
                dur = float(d.get("source_end", 0.0)) - float(d.get("source_start", 0.0))
            if dur > 0.0:
                durations.append(dur)
    elif timed_segments:
        durations = [et - st for _, _, st, et in timed_segments if et > st]

    if durations:
        avg_dur = sum(durations) / len(durations)
        spine_score = min(100.0, max(40.0, (avg_dur / 25.0) * 100.0))
        short_clips = sum(1 for d in durations if d < 4.0)
        if len(durations) > 0 and (short_clips / len(durations)) > 0.20:
            jitter_penalty = ((short_clips / len(durations)) - 0.20) * 50.0
            spine_score -= jitter_penalty
            discontinuity_events.append(
                f"Visual spine jitter: {short_clips}/{len(durations)} clips are ultra-short (<4s)"
            )
        visual_spine_contiguity_score = max(0.0, min(100.0, round(spine_score, 1)))
    else:
        visual_spine_contiguity_score = 100.0

    # 5b. Repetitive Duplicate Sentence Detection across Segments
    import re
    all_sentences: list[tuple[int, str]] = []
    for i, seg in enumerate(segments):
        t = seg.get("text", "")
        parts = [p.strip() for p in re.split(r'[。！？\n]', t) if len(p.strip()) >= 12]
        for p in parts:
            all_sentences.append((i, p))

    seen_sents: dict[str, list[int]] = {}
    for seg_i, s_text in all_sentences:
        seen_sents.setdefault(s_text, []).append(seg_i)

    dup_sentences = {s: occs for s, occs in seen_sents.items() if len(set(occs)) > 1}
    dup_sentence_penalty = min(50.0, len(dup_sentences) * 20.0)
    if dup_sentences:
        for s_text, occs in list(dup_sentences.items())[:3]:
            discontinuity_events.append(
                f"Repetitive duplicate sentence across segments {occs}: '{s_text[:30]}...'"
            )

    # 6. Composite Narrative Continuity Score
    raw_continuity = (
        0.35 * temporal_monotonicity_score
        + 0.35 * (transition_coherence_ratio * 100.0)
        + 0.15 * (character_entity_thread_ratio * 100.0)
        + 0.15 * visual_spine_contiguity_score
    )
    continuity_score = round(max(0.0, raw_continuity - dup_sentence_penalty), 1)

    passed = (continuity_score >= 70.0) and (backward_jumps == 0) and (len(dup_sentences) == 0)

    details = (
        f"Continuity Score={continuity_score:.1f}/100 "
        f"(Monotonicity={temporal_monotonicity_score:.1f}%, "
        f"Transitions={transition_coherence_ratio * 100:.1f}%, "
        f"Entity Threading={character_entity_thread_ratio * 100:.1f}%, "
        f"Spine Contiguity={visual_spine_contiguity_score:.1f}%, "
        f"Duplicate Sents={len(dup_sentences)})"
    )

    metrics = StoryContinuityMetrics(
        temporal_monotonicity_score=temporal_monotonicity_score,
        transition_coherence_ratio=transition_coherence_ratio,
        character_entity_thread_ratio=character_entity_thread_ratio,
        visual_spine_contiguity_score=visual_spine_contiguity_score,
        continuity_score=continuity_score,
        scene_jump_count=scene_jumps,
        unbridged_jump_count=unbridged_jumps,
        backward_jump_count=backward_jumps,
        discontinuity_events=discontinuity_events,
        details=details,
    )

    dim = DimensionScore(
        name="Storyteller Narrative Continuity",
        score=continuity_score,
        weight=0.12,
        passed=passed,
        details=details,
        findings=discontinuity_events,
    )

    return metrics, dim


def evaluate_deterministic(
    script: dict[str, Any],
    story: dict[str, Any] | None = None,
    scene_index: dict[str, Any] | None = None,
    edit_decisions: list[dict[str, Any]] | None = None,
    timeline_audio_duration: float | None = None,
    subtitles: list[dict[str, Any]] | None = None,
) -> tuple[DeterministicMetrics, list[DimensionScore], EvaluatorTelemetry]:

    """
    Run deterministic evaluation suite on generated script and edit assets.

    Args:
        script: Generated script dict.
        story: Story understanding dict with characters/events.
        scene_index: Scene index with extracted scene timestamps.
        edit_decisions: Edit decisions with video clip & audio alignment info.
        timeline_audio_duration: Actual rendered narration audio duration in seconds.
        subtitles: Raw subtitle lines from subtitles.json.

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
        weight=0.08,
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
    meta_matches = scan_for_meta_commentary(full_text)
    repeated = find_repeated_phrases(full_text, ngram_size=5, min_count=3)

    ttr_score = min(100.0, (lex_metrics["ttr"] / 0.45) * 100.0)
    distinct_score = min(100.0, (lex_metrics["distinct_2"] / 0.90) * 100.0)
    cliche_penalty = min(40.0, len(cliche_matches) * 10.0)
    meta_penalty = min(50.0, len(meta_matches) * 15.0)
    rep_penalty = min(30.0, len(repeated) * 5.0)

    diversity_score = max(0.0, round((ttr_score * 0.5 + distinct_score * 0.5) - cliche_penalty - rep_penalty - meta_penalty, 1))

    diversity_findings = []
    if cliche_matches:
        diversity_findings.append(f"Found {len(cliche_matches)} AI clichés: {cliche_matches[:3]}")
    if meta_matches:
        diversity_findings.append(f"Found {len(meta_matches)} ungrounded viewer meta-commentaries / film jargon: {meta_matches[:3]}")
    if repeated:
        diversity_findings.append(f"Found {len(repeated)} repeated 5-grams (≥3x): {repeated[:3]}")

    if len(meta_matches) >= 3:
        hard_failures.append(f"Excessive ungrounded viewer meta-commentary: {len(meta_matches)} occurrences detected (e.g. {meta_matches[:2]})")

    diversity_dim = DimensionScore(
        name="Lexical Diversity & Cliché Avoidance",
        score=diversity_score,
        weight=0.08,
        passed=diversity_score >= 70.0 and len(meta_matches) < 3,
        details=f"TTR={lex_metrics['ttr']:.2f}, Distinct-2={lex_metrics['distinct_2']:.2f}, Clichés={len(cliche_matches)}, Meta={len(meta_matches)}",
        findings=diversity_findings,
    )

    # 3. Evidence Grounding & Temporal Sequence (7% weight)
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
        weight=0.07,
        passed=evidence_score >= 75.0,
        details=f"Evidence Grounding={grounded_ratio * 100:.1f}%, Chronological={chronological_pass}",
        findings=evidence_findings,
    )

    # 4. Cross-Modal Audio-Visual Coupling & Timeline Coverage (15% weight)
    # Checks duration drift, shot duration distribution, timeline span, anti-looping, visual character matching, and AV semantic phase alignment
    max_shot_duration = 0.0

    total_drift = 0.0
    matched_characters = 0
    total_character_checks = 0
    duplicate_clip_loops = 0
    timeline_coverage_ratio = 1.0
    tot_src_dur = scene_index.get("duration_seconds", 300.0) if scene_index else 300.0

    known_characters = [c.get("name") for c in story.get("characters", []) if isinstance(c, dict) and c.get("name")] if story else []

    if edit_decisions:
        for i, dec in enumerate(edit_decisions):
            dur = dec.get("duration", 0.0)
            max_shot_duration = max(max_shot_duration, dur)
            # Drift between video clip and audio track
            v_dur = dec.get("source_end", 0.0) - dec.get("source_start", 0.0)
            if v_dur > 0 and dur > 0:
                total_drift += abs(v_dur - dur)

            # Check consecutive duplicate/looping clip starts
            if i > 0:
                prev_st = edit_decisions[i - 1].get("source_start", 0.0)
                curr_st = dec.get("source_start", 0.0)
                if abs(curr_st - prev_st) < 5.0:
                    duplicate_clip_loops += 1

        if len(edit_decisions) >= 3:
            min_src = min(d.get("source_start", 0.0) for d in edit_decisions)
            max_src = max(d.get("source_end", 0.0) for d in edit_decisions)
            if tot_src_dur > 0:
                timeline_coverage_ratio = min(1.0, (max_src - min_src) / tot_src_dur)

    # Run Cross-Modal AV Semantic Alignment Audit
    av_sem_score, phase_mismatches, phase_findings = check_av_semantic_alignment(
        segments=segments,
        edit_decisions=edit_decisions or [],
        total_source_duration=tot_src_dur,
        movie_title=target_title,
        scene_index=scene_index,
        story_understanding=story,
    )

    # Compute content-grounded semantic match score
    scene_content_match_score = 100.0
    low_match_segments: list[str] = []
    if edit_decisions and scene_index and scene_index.get("scenes"):
        from src.media.scene_matcher import build_semantic_scene_index, score_scene_for_narration
        sem_idx = build_semantic_scene_index(scene_index, story)
        content_scores = []
        for dec in edit_decisions:
            st = dec.get("source_start", 0.0)
            txt = dec.get("text", "")
            if not txt:
                continue
            sc = next((s for s in sem_idx.scenes if s.start_seconds <= st <= s.end_seconds), None)
            if sc is None and sem_idx.scenes:
                sc = min(sem_idx.scenes, key=lambda s: abs(s.start_seconds - st))
            if sc:
                sc_score = score_scene_for_narration(
                    scene=sc,
                    narration_text=txt,
                    target_timestamp=st,
                    total_duration=tot_src_dur,
                )
                content_scores.append(sc_score)
                if sc_score < 0.20:
                    low_match_segments.append(dec.get("segment_id", f"clip-{st:.1f}"))

        if content_scores:
            avg_content_score = sum(content_scores) / len(content_scores)
            scene_content_match_score = round(min(100.0, (avg_content_score / 0.50) * 100.0), 1)

    has_any_scene_char_tags = any(s.get("characters") for s in scenes)
    for seg in segments:
        text = seg.get("text", "")
        supporting = seg.get("supporting_scenes", [])
        # Check if characters mentioned in text appear in supporting scenes
        for char in known_characters:
            if char in text:
                if supporting and scenes:
                    start_s = supporting[0].get("start_seconds", 0.0)
                    end_s = supporting[0].get("end_seconds", start_s)
                    matching_scenes = [
                        s for s in scenes
                        if s.get("start_seconds", 0) <= end_s and s.get("end_seconds", 0) >= start_s
                    ]
                    if matching_scenes:
                        total_character_checks += 1
                        in_char_tags = any(
                            char.lower() in [c.lower() for c in s.get("characters", [])]
                            for s in matching_scenes
                        )
                        in_transcript = any(
                            char.lower() in s.get("transcript_text", "").lower()
                            for s in matching_scenes
                        )
                        if in_char_tags or in_transcript or not has_any_scene_char_tags:
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
    if duplicate_clip_loops > 0:
        coupling_score -= min(40.0, duplicate_clip_loops * 15.0)
        hard_failures.append(f"Visual repetition defect: {duplicate_clip_loops} consecutive looping/duplicate source clip transitions detected")
    if timeline_coverage_ratio < 0.35 and tot_src_dur > 180.0:
        coupling_score -= (0.35 - timeline_coverage_ratio) * 40.0
    if phase_mismatches > 0:
        coupling_score -= min(60.0, phase_mismatches * 25.0)
        hard_failures.append(f"Severe AV semantic misalignment: {phase_mismatches} narrative segments contradict video footage dialogue and activity")
    if scene_content_match_score < 40.0:
        coupling_score -= min(25.0, (40.0 - scene_content_match_score) * 0.5)

    from src.eval.clip_verifier import verify_all_clips
    clip_report = verify_all_clips(
        edit_decisions=edit_decisions or [],
        subtitles=subtitles,
        scene_index=scene_index,
        story=story,
        total_source_duration=tot_src_dur,
    )

    if clip_report.failed_clips > 0:
        coupling_score = max(0.0, coupling_score - min(30.0, (100.0 - clip_report.clip_pass_rate) * 0.5))

    coupling_score = max(0.0, round(coupling_score, 1))

    coupling_findings = []
    if total_drift > 0.5:
        coupling_findings.append(f"Audio-video duration drift: {total_drift:.2f}s")
    if not dynamic_pacing_pass:
        coupling_findings.append(f"Shot duration exceeds dynamic threshold: {max_shot_duration:.1f}s > 35s")
    if duplicate_clip_loops > 0:
        coupling_findings.append(f"Detected {duplicate_clip_loops} duplicate/looping clip transitions")
    if timeline_coverage_ratio < 0.40 and tot_src_dur > 180.0:
        coupling_findings.append(f"Low timeline coverage: spans {timeline_coverage_ratio * 100:.1f}% of movie runtime")
    if scene_content_match_score < 60.0 and edit_decisions:
        coupling_findings.append(f"Low visual scene content match score: {scene_content_match_score:.1f}/100 ({len(low_match_segments)} clips with low correlation)")
    coupling_findings.extend(phase_findings)
    if clip_report.discrepancies:
        coupling_findings.extend(clip_report.discrepancies[:5])

    av_coupling_metrics = AudioVideoCouplingMetrics(
        av_duration_drift_seconds=round(total_drift, 3),
        character_visual_alignment_ratio=round(char_alignment_ratio, 3),
        max_shot_duration_seconds=round(max_shot_duration, 2),
        dynamic_pacing_pass=dynamic_pacing_pass,
        av_semantic_alignment_score=av_sem_score,
        scene_content_match_score=scene_content_match_score,
        low_match_segments=low_match_segments,
        phase_mismatch_count=phase_mismatches,
        coupling_score=coupling_score,
        details=f"AV Semantic Alignment={av_sem_score:.1f}%, Content Match={scene_content_match_score:.1f}%, Drift={total_drift:.2f}s, Visual Grounding={char_alignment_ratio * 100:.1f}%, Timeline Coverage={timeline_coverage_ratio * 100:.1f}%, Loops={duplicate_clip_loops}, Clip Pass Rate={clip_report.clip_pass_rate:.1f}%",
    )

    av_coupling_dim = DimensionScore(
        name="Cross-Modal Audio-Visual Coupling",
        score=coupling_score,
        weight=0.15,
        passed=coupling_score >= 70.0 and duplicate_clip_loops == 0 and phase_mismatches == 0 and clip_report.clip_pass_rate >= 80.0,
        details=av_coupling_metrics.details,
        findings=coupling_findings,
    )

    # 5. Storyteller Narrative Continuity (12% weight)
    continuity_metrics, continuity_dim = calculate_storyteller_continuity(
        segments=segments,
        story=story,
        edit_decisions=edit_decisions,
        total_source_duration=tot_src_dur,
    )
    if continuity_metrics.backward_jump_count >= 2:
        hard_failures.append(
            f"Severe narrative discontinuity: {continuity_metrics.backward_jump_count} backward timeline regressions detected in storyteller sequence"
        )

    # 6. Speaking Rate and Pacing (5% weight)
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
        meta_commentary_matches=meta_matches,
        repeated_phrases=repeated,
        speaking_rate_chars_per_min=speaking_rate,
        evidence_grounded_ratio=round(grounded_ratio, 3),
        chronological_order_pass=chronological_pass,
        hard_failures=hard_failures,
        av_coupling=av_coupling_metrics,
        continuity=continuity_metrics,
        clip_verification=clip_report,
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

    return metrics, [movie_id_dim, clean_dim, diversity_dim, evidence_dim, continuity_dim, av_coupling_dim, pacing_dim], telemetry
