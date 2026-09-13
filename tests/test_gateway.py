"""Tests for LLM Gateway."""
from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock

from src.ai.gateway import LLMGateway, SemanticTask, GatewayResponse


def _make_gateway(config=None):
    """Create a gateway with test config."""
    return LLMGateway(config or {"llm": {"provider": "google", "api_key": "test"}})


def _register_test_task(gateway):
    """Register a test task."""
    task = SemanticTask(
        name="test_task",
        description="A test task",
        reason_not_deterministic="Testing requires judgment",
        model_tier="default",
        max_input_tokens=1000,
        max_output_tokens=500,
        max_cost_usd=0.10,
    )
    gateway.register_task(task)
    return task


def test_unregistered_task_raises():
    """Test that invoking an unregistered task raises ValueError."""
    gw = _make_gateway()
    with pytest.raises(ValueError, match="Unregistered semantic task"):
        gw.invoke(task_name="nonexistent_task", prompt="hello")


def test_budget_exceeded():
    """Test that exceeding budget raises ValueError."""
    gw = _make_gateway({"llm": {"max_cost_per_project_usd": 0.0}})
    _register_test_task(gw)
    with pytest.raises(ValueError, match="budget exceeded"):
        gw.invoke(task_name="test_task", prompt="hello")


def test_cache_hit(tmp_path):
    """Test that cache hits return cached responses."""
    gw = _make_gateway({
        "llm": {
            "provider": "google",
            "api_key": "test",
            "cache_enabled": True,
            "cache_dir": str(tmp_path / "cache"),
        }
    })
    _register_test_task(gw)

    # Manually seed the cache
    import hashlib
    import json
    prompt = "test prompt"
    cache_key = hashlib.sha256(f"test_task:{prompt}".encode()).hexdigest()
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / f"{cache_key}.json").write_text(
        json.dumps({"content": "cached response", "timestamp": "2025-01-01"}),
        encoding="utf-8",
    )

    response = gw.invoke(task_name="test_task", prompt=prompt)
    assert response.cache_hit is True
    assert response.content == "cached response"


def test_cache_hit_json_parsing(tmp_path):
    """Test that cache hits parse JSON payload into parsed field."""
    gw = _make_gateway({
        "llm": {
            "provider": "google",
            "api_key": "test",
            "cache_enabled": True,
            "cache_dir": str(tmp_path / "cache"),
        }
    })
    _register_test_task(gw)

    import hashlib
    import json
    prompt = "json prompt"
    cache_key = hashlib.sha256(f"test_task:{prompt}".encode()).hexdigest()
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    payload = json.dumps({"text": "测试旁白", "confidence": 0.95})
    (cache_dir / f"{cache_key}.json").write_text(
        json.dumps({"content": payload, "timestamp": "2025-01-01"}),
        encoding="utf-8",
    )

    response = gw.invoke(task_name="test_task", prompt=prompt)
    assert response.cache_hit is True
    assert response.parsed == {"text": "测试旁白", "confidence": 0.95}


def test_telemetry_tracking():
    """Test that telemetry is properly tracked."""
    gw = _make_gateway()
    _register_test_task(gw)
    telemetry = gw.get_telemetry()
    assert telemetry["total_invocations"] == 0
    assert telemetry["total_cost_usd"] == 0.0
    assert telemetry["cache_hit_rate"] == 0.0


def test_register_task():
    """Test that register_task adds the task to the registry."""
    gw = _make_gateway()
    task = _register_test_task(gw)
    assert "test_task" in gw._tasks
    assert gw._tasks["test_task"].name == "test_task"


def test_estimate_cost():
    """Test cost estimation for known models."""
    gw = _make_gateway()
    cost = gw._estimate_cost("gemini-2.0-flash", 1000, 500)
    assert cost > 0
    # Input: 1000/1M * 0.10 = 0.0001
    # Output: 500/1M * 0.40 = 0.0002
    assert abs(cost - 0.0003) < 0.0001
