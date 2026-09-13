# Project-specific rules for Movie Commentary Autopilot

## Deterministic-First Policy

This project enforces a strict deterministic-first execution policy. Every workflow step must
first answer: "Can this task be completed reliably by a shell command, script, library, API,
or conventional code?" If yes, the deterministic implementation is REQUIRED.

### Deterministic worker contracts
- All media processing uses FFmpeg/FFprobe
- Scene detection uses PySceneDetect
- Subtitle parsing uses pysrt/webvtt-py
- Schema validation uses Pydantic
- All file hashing uses hashlib SHA-256

### LLM gateway rules
- ALL LLM calls MUST go through src/ai/gateway.py
- Each call must declare: why deterministic processing is insufficient, the smallest context needed, a structured output schema, and a token budget
- No direct model provider calls from any module

## Code Standards
- Python 3.11+
- Type hints on all functions
- Pydantic models for all data entities
- Docstrings on all public functions
- Tests for all deterministic workers
