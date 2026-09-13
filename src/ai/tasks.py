"""
Registry of approved semantic tasks for the LLM gateway.

Each task declares:
- Why deterministic processing is insufficient
- The smallest context required
- A structured output schema
- A token and cost budget
- Whether results can be cached

This module also provides a convenience function to register all standard tasks.
"""

from src.ai.gateway import LLMGateway, SemanticTask


# ============================================================================
# Story Understanding Tasks
# ============================================================================

TASK_SCENE_DESCRIPTION = SemanticTask(
    name="scene_description",
    description="Generate a visual description of a scene from keyframe images",
    reason_not_deterministic=(
        "Visual understanding and natural language description of scene content "
        "requires multimodal interpretation that cannot be done deterministically"
    ),
    model_tier="cheap",
    max_input_tokens=2000,
    max_output_tokens=500,
    max_cost_usd=0.01,
    cacheable=True,
)

TASK_LOCAL_SUMMARY = SemanticTask(
    name="local_summary",
    description="Summarize a bounded sequence of scenes into structured events",
    reason_not_deterministic=(
        "Identifying narrative events, character motivations, and causal relationships "
        "from transcript and scene data requires semantic understanding"
    ),
    model_tier="default",
    max_input_tokens=4000,
    max_output_tokens=2000,
    max_cost_usd=0.05,
    cacheable=True,
)

TASK_CHARACTER_EXTRACTION = SemanticTask(
    name="character_extraction",
    description="Extract character names, aliases, and relationships from transcript data",
    reason_not_deterministic=(
        "Identifying characters, resolving name variations, and understanding "
        "relationships requires language understanding"
    ),
    model_tier="default",
    max_input_tokens=6000,
    max_output_tokens=2000,
    max_cost_usd=0.05,
    cacheable=True,
)

TASK_STORY_SYNTHESIS = SemanticTask(
    name="story_synthesis",
    description="Synthesize local summaries into a global story understanding",
    reason_not_deterministic=(
        "Reconstructing the full narrative arc, identifying themes, and resolving "
        "ambiguities across the entire movie requires complex reasoning"
    ),
    model_tier="strong",
    max_input_tokens=8000,
    max_output_tokens=4000,
    max_cost_usd=0.20,
    cacheable=True,
)

# ============================================================================
# Script Generation Tasks
# ============================================================================

TASK_SCRIPT_GENERATION = SemanticTask(
    name="script_generation",
    description="Generate a narration script segment from story events and scene data",
    reason_not_deterministic=(
        "Writing engaging, original commentary with editorial voice, cultural context, "
        "and natural language flow requires creative generation"
    ),
    model_tier="strong",
    max_input_tokens=8000,
    max_output_tokens=4000,
    max_cost_usd=0.25,
    cacheable=True,
)

TASK_HOOK_GENERATION = SemanticTask(
    name="hook_generation",
    description="Generate a 30-60 second opening hook for the commentary video",
    reason_not_deterministic=(
        "Creating an engaging hook that captures attention requires creative writing "
        "and understanding of audience psychology"
    ),
    model_tier="strong",
    max_input_tokens=4000,
    max_output_tokens=1000,
    max_cost_usd=0.10,
    cacheable=True,
)

TASK_CONCLUSION_GENERATION = SemanticTask(
    name="conclusion_generation",
    description="Generate the concluding segment with assessment and interpretation",
    reason_not_deterministic=(
        "Writing a meaningful conclusion with original analysis and editorial "
        "perspective requires creative synthesis"
    ),
    model_tier="strong",
    max_input_tokens=4000,
    max_output_tokens=1500,
    max_cost_usd=0.10,
    cacheable=True,
)

# ============================================================================
# Verification Tasks
# ============================================================================

TASK_FACTUAL_VERIFICATION = SemanticTask(
    name="factual_verification",
    description="Verify script claims against source evidence",
    reason_not_deterministic=(
        "Judging whether a narration claim is factually supported by evidence requires "
        "semantic comparison between the claim and the source material"
    ),
    model_tier="default",
    max_input_tokens=4000,
    max_output_tokens=2000,
    max_cost_usd=0.05,
    cacheable=True,
)

TASK_QUALITY_EVALUATION = SemanticTask(
    name="quality_evaluation",
    description="Evaluate script quality: coherence, tone, commentary density, naturalness",
    reason_not_deterministic=(
        "Assessing editorial quality, naturalness, engagement, and distinctiveness "
        "requires subjective judgment that deterministic methods cannot provide"
    ),
    model_tier="default",
    max_input_tokens=6000,
    max_output_tokens=2000,
    max_cost_usd=0.05,
    cacheable=False,  # Quality eval may vary
)

# ============================================================================
# Metadata Tasks (for later milestones)
# ============================================================================

TASK_TITLE_GENERATION = SemanticTask(
    name="title_generation",
    description="Generate three title candidates for the YouTube video",
    reason_not_deterministic=(
        "Creating clickable, accurate, and engaging titles requires creative language "
        "generation with audience awareness"
    ),
    model_tier="default",
    max_input_tokens=2000,
    max_output_tokens=500,
    max_cost_usd=0.02,
    cacheable=True,
)

TASK_DESCRIPTION_GENERATION = SemanticTask(
    name="description_generation",
    description="Generate YouTube video description with summary and disclosures",
    reason_not_deterministic=(
        "Writing an informative description that summarizes the commentary "
        "and includes appropriate disclosures requires language generation"
    ),
    model_tier="cheap",
    max_input_tokens=2000,
    max_output_tokens=1000,
    max_cost_usd=0.02,
    cacheable=True,
)


# All registered tasks
ALL_TASKS = [
    TASK_SCENE_DESCRIPTION,
    TASK_LOCAL_SUMMARY,
    TASK_CHARACTER_EXTRACTION,
    TASK_STORY_SYNTHESIS,
    TASK_SCRIPT_GENERATION,
    TASK_HOOK_GENERATION,
    TASK_CONCLUSION_GENERATION,
    TASK_FACTUAL_VERIFICATION,
    TASK_QUALITY_EVALUATION,
    TASK_TITLE_GENERATION,
    TASK_DESCRIPTION_GENERATION,
]


def register_all_tasks(gateway: LLMGateway) -> None:
    """
    Register all standard semantic tasks with the gateway.

    Args:
        gateway: The LLM gateway instance.
    """
    for task in ALL_TASKS:
        gateway.register_task(task)
