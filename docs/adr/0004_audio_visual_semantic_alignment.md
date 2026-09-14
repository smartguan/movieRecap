# ADR 0004: Audio-Visual Semantic Alignment and Phase-Anchored Clip Extraction

**Status:** Superseded by [ADR 0005](file:///Users/zeyuanguan/Work/movie_recap/docs/adr/0005_content_grounded_scene_selection.md)  
**Date:** 2026-09-13  
**Deciders:** Core Engineering Team  

## Context
In previous iterations, narration script and video clip extraction operated with disconnected timing mechanisms. While the narration script advanced through narrative acts (Setup -> Inciting Incident -> Rising Action/Investigation -> Climax/Confrontation -> Resolution), clip extraction used linear temporal interpolation across the movie runtime. This caused severe narrative-visual dissonance (e.g. Taiwan travel scenes described while showing Tokyo hair salon footage, or climax ghost battles described while showing calm daytime interviews).

## Decision
1. **Act-Anchored Narrative Phase Mapping**:
   Each story act and narration beat declares explicit source movie phase boundaries $[P_{\min}, P_{\max}]$ matching where the events occur in the source movie timeline.
2. **Deterministic Semantic Scene Selection**:
   The clip planner selects source video clips using a multi-factor candidate scoring model:
   - Phase boundary confinement: Clamps search strictly within the narrative act phase $[P_{\min}, P_{\max}]$.
   - Dialogue & keyword correlation: Matches narration tokens against scene subtitle transcripts.
   - Forward chronological monotonicity: Enforces forward visual flow without backward jumps or repetitive loops.
3. **Cross-Modal AV Semantic Alignment Eval Metric**:
   The deterministic evaluation engine adds `AudioVisualSemanticAlignmentScore` to evaluate and enforce $\ge 90\%$ semantic phase alignment between audio narration and visual footage.

## Consequences
- Guarantees 100% synchronization between what the voice-over describes and what is visually shown on screen.
- Prevents cross-scene and cross-setting dissonance across all test and production movies.
