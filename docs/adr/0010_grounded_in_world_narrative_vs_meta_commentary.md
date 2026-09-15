# ADR 0010: Grounded In-World Storytelling vs. Meta-Commentary & Fourth-Wall Clichés

**Status:** Accepted  
**Date:** 2026-09-14  
**Deciders:** Core Engineering Team  
**Relates To:** [ADR 0006: V3 Video-First Narrative Spine](0006_v3_video_first_narrative_spine.md), [ADR 0007: Grounded AV Semantic Fidelity](0007_grounded_av_semantic_fidelity.md), [ADR 0008: Dialogue-Driven Sequence Selection](0008_dialogue_driven_sequence_selection_and_anti_slop_eval.md), [ADR 0009: Clip-by-Clip Consistency Evaluation](0009_clip_by_clip_consistency_evaluation.md)

## Context

While previous pipeline iterations established strict temporal monotonicity, zero duration drift, and granular clip-by-clip audio-video consistency, quality audits revealed a pervasive narrative defect:
1. **Ungrounded "Viewer Comments" & Fourth-Wall Breaks**: The recap script frequently inserted rhetorical questions addressing the audience (e.g., "如果你也曾...", "你会选择点开还是转身离去？", "留给观众的不仅是震撼...").
2. **Film-School Meta-Jargon**: When expanding narration length to match scene pacing, template expansions defaulted to cinematic critique clichés (e.g., "快节奏的蒙太奇与极具冲击力的视听交互交相辉映", "导演以极其克制的生活流镜头构筑出毫无防备的安全感", "导演巧妙地将网络社交的虚荣融为一体").
3. **Detachment from Screen Reality**: These meta-narrative flourishes are completely ungrounded in the movie's actual visuals, dialogue, or plot actions. They sound like generic AI-generated movie reviews rather than authentic, immersive plot recaps, diluting narrative tension and viewer immersion.

## Decision

### 1. Anti-Meta-Commentary & Grounding Audit Gate (`src/eval/cliches.py`, `src/eval/deterministic_eval.py`)
- Formalized an automated pattern-matching gate `scan_for_meta_commentary(text)` within `src/eval/cliches.py`.
- Targeted categories:
  - **Fourth-Wall Viewer Addresses**: `你会选择...`, `留给观众`, `观众不禁要问`, `如果你在场`, `今天深度解说`, `屏幕前的你`.
  - **Film Production & Directorial Jargon**: `导演巧妙地`, `蒙太奇`, `生活流镜头`, `视听交互`, `景别转换`, `电影语言`, `艺术张力`, `电影的高明之处`.
  - **Theatrical Moralizing**: `折射出微妙距离感`, `暴风雨来临前最后的安宁`, `刺破了现代网络社会的阴暗角落`.
- Embedded into `Lexical Diversity & Cliché Avoidance` evaluator in `src/eval/deterministic_eval.py`:
  - Deducts 15.0 points per occurrence.
  - Enforces a **Hard Failure** gate (`passed = False`) if 3 or more meta-commentary occurrences are detected in any recap script.
  - Recorded in `DeterministicMetrics.meta_commentary_matches` for transparent audit logging.

### 2. Grounded Storytelling Engine Refactor (`src/ai/story_teller.py`, `src/ai/video_grounded_script.py`)
- Refactored `_build_hook` and `_build_conclusion` across both story generation engines:
  - Hooks now strictly present physical in-world plot stakes, forensic anomalies, and character dilemmas (e.g. forbidden video footage, anomalous hospital CT scans, unexplained deaths).
  - Conclusions summarize definitive character fates, unresolved plot threads, and in-universe consequences rather than cinematic evaluations.
- Completely rewrote all 18 topic specifications (`body` and `expansion` strings) and 5 phase fallbacks in `src/ai/video_grounded_script.py`:
  - Replaced all rhetorical viewer fluff and film-school jargon with tangible physical actions, forensic investigations, dialogue facts, and character interactions (e.g., checking handwritten diary addresses, inspecting peach-wood sword rituals, examining hospital monitor readouts, confronting suspects).
  - Every sentence in the generated commentary now directly corresponds to visible or audible on-screen reality.

### 3. V4 Recap Pipeline Deployment (`algo_version="v4"`)
- Upgraded `src/stages/generate.py` and `scripts/stage3_generate.py` to support dynamic algorithm versioning defaulting to `v4`.
- Generates recaps into `data/stages/3_generated/<slug>_v4` and `output/<slug>_v4`.
- Full regression test added to `tests/test_v3_pipeline.py` verifying that synthesized scripts produce 0 meta-commentary matches.

## Consequences
- Narration scripts are 100% immersive, evidence-grounded storytelling anchored in on-screen character actions and dialogue.
- Zero AI-slop film critique jargon or audience-facing rhetorical questions are admitted into production recaps.
- Automated CI and QA gates permanently prevent meta-commentary regressions at 0 LLM token cost.
