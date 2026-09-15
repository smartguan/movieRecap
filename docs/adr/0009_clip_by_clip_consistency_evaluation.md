# ADR 0009: Clip-by-Clip Audio & Video Consistency Evaluation Engine

**Status:** Accepted  
**Date:** 2026-09-14  
**Deciders:** Core Engineering Team  
**Relates To:** [ADR 0007: Grounded AV Semantic Fidelity](0007_grounded_av_semantic_fidelity.md), [ADR 0008: Dialogue-Driven Sequence Selection](0008_dialogue_driven_sequence_selection_and_anti_slop_eval.md)

## Context
While high-level macro telemetry (such as total duration drift, lexical diversity, and continuity scores) provides overall quality safeguards, subtle discrepancies between narration and footage can still slip through if evaluations are aggregated only across whole-script averages. For instance:
- A single clip describing haircutting placed over kitchen cooking footage could go unnoticed if the rest of the recap scores highly.
- Post-dinner narrative bridges (e.g. referencing the prior hair salon before moving into apartment dining) could trigger false-positive conflict alarms if activity detection does not differentiate between narrative transitions and current-scene assertions.
- Engineers and creators reviewing recap outputs need immediate, granular visibility into every clip: the exact source dialogue/subtitles, the voiceover text, detected activity domains, duration drifts, and individual pass/fail verdicts.

## Decision

### 1. Dedicated Clip Verifier (`src/eval/clip_verifier.py`)
- Implemented a 0-token deterministic clip-by-clip verification engine.
- For every clip in an edit decision list (`edit_decisions` or `edit_plan.json`):
  - **Subtitle Window Extraction**: Dynamically queries `subtitles.json` (and falls back to `scene_index.json`) to extract all dialogue lines falling within `[source_start, source_end]`.
  - **Cross-Modal Activity Conflict Detection**: Compares narration activity cues against visual/subtitle activity cues using `ACTIVITY_DOMAINS`.
  - **Transition-Aware Conflict Resolution**: Ensures that transitional context (e.g. mentioning a prior setting while introducing a new setting) does not trigger false positives if the footage activity is also affirmed by the narration.
  - **Dialogue Semantic Alignment**: Evaluates cross-modal dialogue and topic correlation using `score_scene_for_narration` and concept mapping.
  - **Timing & Drift Guardrails**: Audits duration drift between video cut lengths and audio durations ($\le 1.0$s tolerance).
  - **Chronological Monotonicity**: Flags backward timeline jumps $> 45$s.

### 2. Schema Models & Data Integration (`src/eval/models.py`)
- Created `ClipVerificationResult` to capture per-clip telemetry:
  - `clip_index`, `segment_id`, `recap_timeline`, `source_movie_window`, `source_start`, `source_end`, `duration`, `narration_text`, `video_subtitles`, `video_activity`, `audio_activity`, `semantic_match_score`, `passed`, `findings`, `verdict_details`.
- Created `ClipByClipReport` to aggregate:
  - `total_clips`, `passed_clips`, `failed_clips`, `clip_pass_rate`, `average_semantic_score`, `discrepancies`, and `formatted_table_markdown`.
- Embedded `clip_verification` directly in `DeterministicMetrics` and `AntiSlopReport`.

### 3. Automated Markdown Audit Table Rendering (`src/eval/slop_detector.py`)
- Enhanced `SlopDetector.format_markdown_report` to automatically append a full `## 🔍 Clip-by-Clip Audio & Video Consistency Audit` section.
- Renders the complete, clean Markdown table displaying:
  `| # | Recap Timeline | Source Movie Window | On-Screen Video Content & Subtitles | Audio Narration (Voiceover) | Consistency Verdict |`
- Enables one-glance auditing of every clip in `recap_quality_report.md` without needing external review tools.

### 4. Stage 3 Asset Persistence (`src/stages/generate.py`)
- In addition to `recap_quality_report.md` and `recap_quality_report.json`, Stage 3 now persists `clip_verification.json` in the generated stage directory for programmatic consumption by web dashboards and downstream tooling.

## Consequences
- Every recap generated or audited by the pipeline receives a 100% transparent, granular clip-by-clip audit table.
- Mismatched activities, desyncs, or dialogue contradictions are caught at the individual clip level before publishing.
- All verifications execute deterministically at 0 LLM tokens ($0.00 cost).
