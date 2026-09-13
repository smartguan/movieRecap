# Project-specific rules for Movie Commentary Autopilot

## Spec-Driven Development & Living Architecture

This project is spec-driven. All core business rules, entity models, state transitions, and worker contracts are specified in `docs/product_spec.md` and related documents.

### Rules for Agent Sessions
1. **Consult Specs First**: Before refactoring or implementing features, consult the specs in `docs/` rather than reverse-engineering logic from code.
2. **Synchronous Spec Evolution**: When domain models, states, or pipeline requirements evolve, update `docs/product_spec.md` or create an Architectural Decision Record in `docs/adr/` in the same commit.
3. **Deprecation Handling**: Never silently change business logic. Mark superseded sections with deprecation notices and references to newer requirements.

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
- ALL LLM calls MUST go through `src/ai/gateway.py`
- Each call must declare: why deterministic processing is insufficient, the smallest context needed, a structured output schema, and a token budget
- No direct model provider calls from any module

## Code Standards
- Python 3.9+ compatible (using `from __future__ import annotations`)
- Type hints on all functions
- Pydantic models for all data entities
- Docstrings on all public functions
- Tests for all deterministic workers
