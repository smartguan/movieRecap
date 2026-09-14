# ADR 0007: Generic Audio-Visual Semantic Grounding and Multi-Dimension Quality Gating

**Status:** Accepted  
**Date:** 2026-09-14  
**Deciders:** Core Engineering Team  
**Relates To:** [ADR 0004: Audio-Visual Semantic Alignment](file:///Users/zeyuanguan/Work/movie_recap/docs/adr/0004_audio_visual_semantic_alignment.md), [ADR 0006: V3 Video-First Narrative Spine](file:///Users/zeyuanguan/Work/movie_recap/docs/adr/0006_v3_video_first_narrative_spine.md)

## Context
During manual review of rendered video assets, a severe semantic contradiction was identified:
- **Audio commentary**: Stated that coworkers were having a lunch break in a salon lounge chatting about internet rumors (*"午休时分，同伴们聚在休息室里享用午餐，边吃边聊起网络八卦"*).
- **Video footage**: Showed two girls at home in an apartment kitchen cooking and eating a hot dinner at night (`サイコーンでしょ食べる... 今日はねかなりいい感じで出来ました いただきます`).

Despite this severe disconnect, the evaluation suite awarded the recap **91.8/100 (Tier S)**. Root-cause analysis revealed critical flaws in both generation and evaluation:
1. **Script Generation Flaw**: Script generation relied on hardcoded movie dictionaries or generic static templates rather than extracting and inspecting the actual sequence dialogue (`subtitles.json`) and scene events.
2. **Evaluation Blindness**:
   - `check_av_semantic_alignment` relied on movie-specific keyword landmarks mapped to massive 15-to-40-minute timestamp windows, allowing a cooking scene at 332s to pass as an Act 1 salon scene (`0-1000s`).
   - Correlation was gated on a tiny 10-word `CONCEPT_MAP` (`死`, `杀`, `鬼`, etc.); non-horror scenes were skipped.
   - Character visual alignment contained a bug (`if in_scene or len(scenes) > 0: matched_characters += 1`) that caused it to always pass at 100%.
   - In `slop_detector.py`, dimension failures were ignored if the overall weighted score was $\ge 70.0$, allowing an AV-contradictory recap to pass with Grade S.
   - The LLM semantic verifier received only 5 high-level story bullets, lacking the actual sequence dialogue.

## Decision

### 1. Generic Activity-Domain Conflict Detection (`src/eval/deterministic_eval.py`)
- Removed all movie-specific title conditionals (`if "诅咒" in title_clean`).
- Introduced generic cross-modal activity and setting incompatibility rules (`ACTIVITY_DOMAINS`):
  - `dining_cooking` (eating, cooking, dinner, meal)
  - `salon_haircut` (haircut, styling, shampoo, salon)
  - `phone_digital` (social media, mobile posts, screen alerts)
  - `travel_transit` (airports, flights, highway driving)
  - `temple_ritual` (shrines, incense shops, paper dolls, altars)
  - `combat_horror` (battles, ghosts, flame combat, destruction)
- Detects mutually exclusive activity conflicts between narration assertions and footage dialogue (e.g. asserting salon haircutting during home cooking dialogue).
- Detects macro chronological phase violations (e.g. final combat placed in the first 25% of runtime, or opening exposition placed in the final 30% of runtime).
- Severe AV semantic contradictions trigger `hard_failures` and immediately fail the `Cross-Modal Audio-Visual Coupling` dimension.

### 2. Strict Critical Dimension Gating (`src/eval/slop_detector.py`)
- Established 5 **Critical Quality Dimensions**:
  1. `Movie Identity & Anti-Contamination`
  2. `Cleanliness & Formatting`
  3. `Evidence & Temporal Alignment`
  4. `Cross-Modal Audio-Visual Coupling`
  5. `Storyteller Narrative Continuity`
- If **ANY** critical dimension fails (`passed: False`), the project **cannot receive Grade S or pass the quality gate**. It is immediately rejected with **Tier F**.

### 3. Truly Generic Sequence-Grounded Narration (`src/ai/video_grounded_script.py`)
- Removed all hardcoded movie dictionaries (`curse_scripts`).
- The V3 synthesizer strictly derives commentary from:
  - Sequence dialogue transcripts (`subtitles.json`).
  - Active character entities and dramatic phases.
  - Preceding scene time gap (adding connective transition lead-ins when gap $\ge 45$s).
- Supports both LLM Gateway synthesis (`TASK_SYNTHESIZE_GROUNDED_NARRATION`) and deterministic dialogue-grounded synthesis.

### 4. Sequence Dialogue Visibility in Semantic Verification (`src/ai/verify.py`)
- `_semantic_verification` now includes the exact sequence dialogue alongside each narration segment, empowering the LLM verifier to flag any AV semantic hallucinations.

## Consequences
- **Zero-Tolerance for Hallucinated Settings**: Audio commentary claiming activities contradicted by on-screen footage will be caught deterministically and blocked before shipping.
- **True Multi-Movie Genericity**: The pipeline operates purely on subtitle transcripts, scene cuts, and character registers without per-movie hardcoding.
- **Reliable Continuous Improvement**: The evaluation suite reproduces inconsistencies objectively, providing a trustworthy signal for future model and prompt iterations.
