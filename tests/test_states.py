"""Tests for workflow state machine."""
from __future__ import annotations

from src.orchestrator.states import (
    ProjectState,
    validate_transition,
    InvalidTransitionError,
    get_next_state,
    VALID_TRANSITIONS,
    TERMINAL_STATES,
)
import pytest


def test_valid_transitions_succeed():
    """Test that valid transitions pass validation."""
    assert validate_transition(ProjectState.RECEIVED, ProjectState.INGESTING)
    assert validate_transition(ProjectState.INGESTING, ProjectState.ANALYZING)
    assert validate_transition(ProjectState.ANALYZING, ProjectState.SCRIPTING)


def test_invalid_transitions_raise():
    """Test that invalid transitions raise InvalidTransitionError."""
    with pytest.raises(InvalidTransitionError):
        validate_transition(ProjectState.RECEIVED, ProjectState.PUBLISHED)
    with pytest.raises(InvalidTransitionError):
        validate_transition(ProjectState.INGESTING, ProjectState.PUBLISHED)


def test_get_next_state_happy_path():
    """Test get_next_state returns the correct next state in the happy path."""
    assert get_next_state(ProjectState.RECEIVED) == ProjectState.INGESTING
    assert get_next_state(ProjectState.INGESTING) == ProjectState.ANALYZING
    assert get_next_state(ProjectState.ANALYZING) == ProjectState.SCRIPTING
    assert get_next_state(ProjectState.SCRIPTING) == ProjectState.GENERATING_AUDIO


def test_terminal_states_have_no_next():
    """Test that terminal states return None for get_next_state."""
    assert get_next_state(ProjectState.PUBLISHED) is None
    assert get_next_state(ProjectState.ARCHIVED) is None


def test_terminal_states_have_no_transitions():
    """Test that terminal states have no valid transitions."""
    for state in TERMINAL_STATES:
        assert VALID_TRANSITIONS.get(state, set()) == set()


def test_error_states_can_retry():
    """Test that FAILED can transition to RETRYING."""
    assert validate_transition(ProjectState.FAILED, ProjectState.RETRYING)


def test_awaiting_approval_can_approve_regenerate_reject():
    """Test the three possible outcomes from approval."""
    assert validate_transition(ProjectState.AWAITING_APPROVAL, ProjectState.APPROVED)
    assert validate_transition(ProjectState.AWAITING_APPROVAL, ProjectState.SCRIPTING)
    assert validate_transition(ProjectState.AWAITING_APPROVAL, ProjectState.ARCHIVED)
