# ADR 0006: V3 Video-First Narrative Spine Engine and Storyteller Continuity Evaluation

**Status:** Accepted  
**Date:** 2026-09-14  
**Deciders:** Core Engineering Team  
**Relates To:** [ADR 0005: Content-Grounded Scene Selection](file:///Users/zeyuanguan/Work/movie_recap/docs/adr/0005_content_grounded_scene_selection.md)

## Context
In previous iterations (v1 and v2), the recap pipeline was **script-first**:
1. An abstract commentary text was synthesized from high-level summaries and event metadata.
2. The clip planner then hunted across hundreds of fragmented shot cuts to find 2–3 isolated snippets matching each paragraph.
3. This produced severe visual fragmentation: viewers watched 6–8 second micro-cuts (slideshow effect) with abrupt jumps, and even with semantic keyword matching, video shots often lacked contiguous narrative progression.

The user proposed a new **video-first narrative-spine** paradigm:
1. Cut the original movie into logical continuous macro-segments.
2. Filter out sideline subplots and atmospheric filler to retain only the primary dramatic spine.
3. Further cluster into key continuous dramatic blocks (~45s–120s each) hitting 1/5 total runtime budget.
4. Synthesize voiceover commentary directly grounded in each continuous video segment's on-screen dialogue and action, with connective transition lead-ins bridging time jumps.
5. Provide continuous evaluation metrics to rigorously measure the "continuity" of the movie storyteller.
6. Preserve the V2 pipeline intact to enable direct side-by-side A/B testing (`algo_version="v2"` vs `algo_version="v3"`).

## Decision

### 1. Deterministic Sequence Clustering (`src/media/sequence_clusterer.py`)
- Groups granular PySceneDetect shot cuts into contiguous **Macro-Sequences** (45s–120s).
- Uses dialogue continuity (subtitle silence gaps $< 4.0$s), character presence, and physical temporal boundaries with 0 LLM tokens.
- Guarantees unbroken contiguous time intervals for every scene block.

### 2. Main Story Filtering & Key Moment Selection (`src/ai/story_filter.py`)
- Distinguishes main story conflict from subplots and dead pauses.
- Scores dramatic intensity and allocates budget across 5 narrative phases (Setup 18%, Inciting Incident 24%, Investigation 28%, Climax 20%, Resolution 10%).
- Selects 16–22 contiguous dramatic sequences hitting the exact 1/5 movie duration target (~18.9 min for a 94.6 min film).

### 3. Video-Grounded Script Synthesis (`src/ai/video_grounded_script.py`)
- Generates commentary directly derived from each selected video sequence.
- Every beat contains:
  1. `Transition Lead-In`: Connective narrative phrasing bridging the temporal/spatial jump from previous scenes.
  2. `Dialogue & Action Narration`: Accurate Mandarin interpretation of on-screen speech and action.
  3. `Duration Synchronization`: Segment length calibrated to speech rate (240 chars/min) matching sequence duration.

### 4. Storyteller Continuity Evaluation Metrics (`src/eval/models.py` & `src/eval/deterministic_eval.py`)
- Added `StoryContinuityMetrics` and a dedicated **Storyteller Narrative Continuity** dimension (12% weight) evaluated deterministically (0 LLM tokens):
  - **Temporal Monotonicity (35%)**: Audits chronological progression; heavily penalizes backward time regressions ($>15$s).
  - **Discourse Transition Coherence (35%)**: Audits scene jumps ($\ge 45$s); verifies presence of bridging transition discourse markers.
  - **Character / Entity Threading (15%)**: Evaluates topic continuity across adjacent narrative beats.
  - **Visual Spine Contiguity (15%)**: Evaluates macro-scene duration stability vs. micro-fragmentation jitter.

### 5. Seamless A/B Swappability (`src/stages/generate.py` & `src/media/clip_planner.py`)
- `run_stage_generate(..., algo_version="v3")` defaults to V3 video-first generation.
- Passing `algo_version="v2"` executes the existing script-first StoryTeller + content-grounded matcher pipeline.
- Both pipelines can be run on the same analyzed source movie for comparative evaluation and benchmarking.

## Consequences
- **Visual Scene Cohesion**: Viewers experience continuous, real-time dramatic scenes rather than disjointed slideshow cuts.
- **Flawless AV Coupling**: Commentary is generated directly from the footage shown on screen.
- **Objective Quality Gate**: The eval suite detects unbridged scene leaps, chronological regressions, and shot jitter before shipping to viewers.
