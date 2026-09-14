# ADR 0005: Content-Grounded Audio-Video Semantic Matching and Evaluation

**Status:** Accepted  
**Date:** 2026-09-14  
**Deciders:** Core Engineering Team  
**Supersedes:** [ADR 0004: Audio-Visual Semantic Alignment](file:///Users/zeyuanguan/Work/movie_recap/docs/adr/0004_audio_visual_semantic_alignment.md)

## Context
In ADR 0004, the pipeline adopted act-anchored temporal boundaries and keyword-based phase checks. However, empirical testing revealed a critical deficiency:
1. Act boundaries in the script generator were interpolated mathematically rather than tied to actual source visual events.
2. The clip planner used geometric proximity to proposed timestamps, ignoring scene subtitles, dialogues, and visual content.
3. The evaluation engine checked only broad phase windows (spanning 10–20 minutes), passing videos with 100% score even when the voiceover described a hair salon while the screen displayed unrelated night footage.

## Decision
1. **Deterministic Scene Semantic Indexing (`src/media/scene_matcher.py`)**:
   - Indexes all scene transcripts, local summaries, key events, and character appearances.
   - Deterministically tokenizes CJK text (unigrams, bigrams, and terms) with 0 LLM tokens.
   - Maintains bilingual concept mapping between Mandarin commentary themes (e.g. 理发, 浴缸, 台湾, 纸人, 神庙, 怨灵) and source subtitle/visual cues.
   - Expands contiguous temporal context windows (±1 scene within $\le 8.0$s) to connect reaction shots with spoken dialogue.

2. **Multi-Signal Affinity Scoring (`score_scene_for_narration`)**:
   Candidate scenes are scored on a composite metric:
   - **Concept & Term Mapping (35%)**: Direct matching of core plot/setting entities.
   - **Transcript & Subtitle Lexical Overlap (25%)**: Exact token intersection with spoken/ASR text.
   - **Local Summary & Narrative Event Overlap (20%)**: Semantic correlation with Stage 2 chunk summaries.
   - **Phase Window Conformity (15%)**: Alignment with narrative act bounds.
   - **Chronological Progression (5%)**: Rewarding forward flow while permitting semantic cuts.

3. **Content-Aware Clip Planning (`src/media/clip_planner.py`)**:
   - Replaces geometric proximity snapping with semantic candidate ranking via `find_best_scenes`.
   - Enforces $8.0$s minimum spacing and anti-looping deduplication.
   - Binds source cuts directly to scenes where the narrated event physically takes place.

4. **Rigorous Content-Grounded Evaluation (`src/eval/deterministic_eval.py`)**:
   - Upgrades `check_av_semantic_alignment` from broad phase checks to direct scene content verification at `source_start`.
   - Adds `scene_content_match_score` (0–100) and `low_match_segments` to `AudioVideoCouplingMetrics`.
   - Penalizes projects with low visual correlation and triggers QA failure if content disconnect is detected.

## Consequences
- Guaranteed visual-narrative coherence: what is narrated in the voiceover is directly shown on screen.
- Elimination of false-positive 100% QA evaluations when audio and video are disjointed.
- Full 100% deterministic operation (0 LLM tokens) with millisecond-speed index lookups.
