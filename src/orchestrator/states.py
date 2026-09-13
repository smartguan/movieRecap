"""
Workflow state definitions and transition rules.

This module defines the state machine for project processing.
All state transitions are deterministic and validated.
"""

from __future__ import annotations
from enum import Enum


class ProjectState(str, Enum):
    """All possible states in the project lifecycle."""

    # Happy path
    RECEIVED = "received"
    INGESTING = "ingesting"
    ANALYZING = "analyzing"
    SCRIPTING = "scripting"
    GENERATING_AUDIO = "generating_audio"
    PLANNING_EDIT = "planning_edit"
    RENDERING = "rendering"
    AUTOMATED_QA = "automated_qa"
    YOUTUBE_PRIVATE = "youtube_private"
    AWAITING_APPROVAL = "awaiting_approval"
    APPROVED = "approved"
    SCHEDULED = "scheduled"
    PUBLISHED = "published"

    # Error / exception states
    RETRYING = "retrying"
    NEEDS_REVIEW = "needs_review"
    FAILED = "failed"
    ARCHIVED = "archived"


# Valid state transitions: from_state -> set of allowed to_states
VALID_TRANSITIONS: dict[ProjectState, set[ProjectState]] = {
    ProjectState.RECEIVED: {
        ProjectState.INGESTING,
        ProjectState.FAILED,
    },
    ProjectState.INGESTING: {
        ProjectState.ANALYZING,
        ProjectState.RETRYING,
        ProjectState.FAILED,
    },
    ProjectState.ANALYZING: {
        ProjectState.SCRIPTING,
        ProjectState.RETRYING,
        ProjectState.NEEDS_REVIEW,
        ProjectState.FAILED,
    },
    ProjectState.SCRIPTING: {
        ProjectState.GENERATING_AUDIO,
        ProjectState.RETRYING,
        ProjectState.NEEDS_REVIEW,
        ProjectState.FAILED,
    },
    ProjectState.GENERATING_AUDIO: {
        ProjectState.PLANNING_EDIT,
        ProjectState.RETRYING,
        ProjectState.FAILED,
    },
    ProjectState.PLANNING_EDIT: {
        ProjectState.RENDERING,
        ProjectState.RETRYING,
        ProjectState.FAILED,
    },
    ProjectState.RENDERING: {
        ProjectState.AUTOMATED_QA,
        ProjectState.RETRYING,
        ProjectState.FAILED,
    },
    ProjectState.AUTOMATED_QA: {
        ProjectState.YOUTUBE_PRIVATE,
        ProjectState.NEEDS_REVIEW,
        ProjectState.RETRYING,
        ProjectState.FAILED,
    },
    ProjectState.YOUTUBE_PRIVATE: {
        ProjectState.AWAITING_APPROVAL,
        ProjectState.RETRYING,
        ProjectState.FAILED,
    },
    ProjectState.AWAITING_APPROVAL: {
        ProjectState.APPROVED,
        ProjectState.SCRIPTING,  # Regenerate
        ProjectState.ARCHIVED,  # Reject
    },
    ProjectState.APPROVED: {
        ProjectState.SCHEDULED,
        ProjectState.PUBLISHED,
        ProjectState.FAILED,
    },
    ProjectState.SCHEDULED: {
        ProjectState.PUBLISHED,
        ProjectState.FAILED,
    },
    ProjectState.PUBLISHED: set(),  # Terminal state

    # Error states
    ProjectState.RETRYING: {
        # Can retry back to any processing state
        ProjectState.INGESTING,
        ProjectState.ANALYZING,
        ProjectState.SCRIPTING,
        ProjectState.GENERATING_AUDIO,
        ProjectState.PLANNING_EDIT,
        ProjectState.RENDERING,
        ProjectState.AUTOMATED_QA,
        ProjectState.YOUTUBE_PRIVATE,
        ProjectState.FAILED,
    },
    ProjectState.NEEDS_REVIEW: {
        ProjectState.ANALYZING,
        ProjectState.SCRIPTING,
        ProjectState.AUTOMATED_QA,
        ProjectState.ARCHIVED,
        ProjectState.FAILED,
    },
    ProjectState.FAILED: {
        ProjectState.ARCHIVED,
        ProjectState.RETRYING,
    },
    ProjectState.ARCHIVED: set(),  # Terminal state
}


# States where the project is actively being processed
PROCESSING_STATES = {
    ProjectState.INGESTING,
    ProjectState.ANALYZING,
    ProjectState.SCRIPTING,
    ProjectState.GENERATING_AUDIO,
    ProjectState.PLANNING_EDIT,
    ProjectState.RENDERING,
    ProjectState.AUTOMATED_QA,
    ProjectState.YOUTUBE_PRIVATE,
}

# Terminal states
TERMINAL_STATES = {
    ProjectState.PUBLISHED,
    ProjectState.ARCHIVED,
}

# States requiring human action
HUMAN_ACTION_STATES = {
    ProjectState.AWAITING_APPROVAL,
    ProjectState.NEEDS_REVIEW,
}


class InvalidTransitionError(Exception):
    """Raised when an invalid state transition is attempted."""

    def __init__(self, from_state: ProjectState, to_state: ProjectState) -> None:
        self.from_state = from_state
        self.to_state = to_state
        super().__init__(
            f"Invalid state transition: {from_state.value} -> {to_state.value}. "
            f"Allowed transitions from {from_state.value}: "
            f"{', '.join(s.value for s in VALID_TRANSITIONS.get(from_state, set()))}"
        )


def validate_transition(from_state: ProjectState, to_state: ProjectState) -> bool:
    """
    Validate whether a state transition is allowed.

    Args:
        from_state: Current project state.
        to_state: Desired next state.

    Returns:
        True if the transition is valid.

    Raises:
        InvalidTransitionError: If the transition is not allowed.
    """
    allowed = VALID_TRANSITIONS.get(from_state, set())
    if to_state not in allowed:
        raise InvalidTransitionError(from_state, to_state)
    return True


def get_next_state(current_state: ProjectState) -> ProjectState | None:
    """
    Get the next state in the happy-path sequence.

    Args:
        current_state: Current project state.

    Returns:
        The next state in the happy path, or None if at a terminal/decision state.
    """
    happy_path = [
        ProjectState.RECEIVED,
        ProjectState.INGESTING,
        ProjectState.ANALYZING,
        ProjectState.SCRIPTING,
        ProjectState.GENERATING_AUDIO,
        ProjectState.PLANNING_EDIT,
        ProjectState.RENDERING,
        ProjectState.AUTOMATED_QA,
        ProjectState.YOUTUBE_PRIVATE,
        ProjectState.AWAITING_APPROVAL,
        ProjectState.APPROVED,
        ProjectState.SCHEDULED,
        ProjectState.PUBLISHED,
    ]
    try:
        idx = happy_path.index(current_state)
        if idx + 1 < len(happy_path):
            return happy_path[idx + 1]
    except ValueError:
        pass
    return None
