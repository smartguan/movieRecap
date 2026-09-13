# Movie Commentary Autopilot

An AI-powered production system that transforms human-supplied movie files into finished YouTube commentary videos. The human retains three responsibilities: supplying the source movie, reviewing the completed video, and approving publication.

## MVP Target

- **Language:** Mandarin (中文)
- **Duration:** 20–30 minute commentary videos
- **Niche:** Korean thriller and drama films
- **Output:** 1080p video with subtitles, branding, and licensed music

## Architecture

The system has two execution planes:

- **Deterministic Plane:** Shell scripts, Python workers, FFmpeg, parsers, schema validators — handles all mechanical work
- **Semantic Plane:** LLM gateway with registered tasks — handles story understanding, commentary writing, and qualitative review

## Quick Start

```bash
# Install dependencies
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Configure
cp config/settings.example.yaml config/settings.yaml
# Edit settings.yaml with your API keys and paths

# Process a movie
mkdir -p data/incoming/my-movie
cp /path/to/movie.mp4 data/incoming/my-movie/source.mp4
python -m src.orchestrator.workflow data/incoming/my-movie
```

## Project Structure

```
├── config/           # Channel profile and system settings
├── src/
│   ├── orchestrator/ # Workflow state machine
│   ├── media/        # Deterministic media workers (ffmpeg, ffprobe)
│   ├── index/        # Scene and transcript indexing
│   ├── ai/           # LLM gateway and semantic tasks
│   ├── models/       # Pydantic data models
│   └── utils/        # Shared utilities
├── scripts/          # Shell scripts for media operations
├── tests/            # Test suite
└── data/             # Runtime data (gitignored)
```

## Workflow States

```
RECEIVED → INGESTING → ANALYZING → SCRIPTING → GENERATING_AUDIO → PLANNING_EDIT →
RENDERING → AUTOMATED_QA → YOUTUBE_PRIVATE → AWAITING_APPROVAL → APPROVED →
SCHEDULED → PUBLISHED
```

## Product Principles

1. AI produces; the human authorizes
2. Evidence-grounded generation
3. Commentary, not mechanical transcription
4. Fail closed
5. No rights circumvention
6. Deterministic rendering
7. Full auditability
8. Deterministic-first and token-efficient

## License

Private — All rights reserved.
