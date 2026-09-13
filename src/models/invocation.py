"""
LLM invocation tracking data models.
"""
from __future__ import annotations
from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, computed_field, Field


class ModelInvocation(BaseModel):
    """Tracking data for a single LLM invocation."""
    invocation_id: str
    task_name: str
    reason: str
    model_name: str
    model_version: str
    prompt_version: str
    input_tokens: int
    output_tokens: int
    estimated_cost_usd: float
    latency_seconds: float
    cache_hit: bool = False
    confidence: Optional[float] = None
    result_hash: str
    evidence_references: List[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class InvocationLedger(BaseModel):
    """A ledger containing all LLM invocations for a project."""
    project_id: str
    invocations: List[ModelInvocation] = Field(default_factory=list)

    @computed_field
    @property
    def total_input_tokens(self) -> int:
        """Total input tokens across all invocations."""
        return sum(inv.input_tokens for inv in self.invocations)

    @computed_field
    @property
    def total_output_tokens(self) -> int:
        """Total output tokens across all invocations."""
        return sum(inv.output_tokens for inv in self.invocations)

    @computed_field
    @property
    def total_cost_usd(self) -> float:
        """Total estimated cost in USD."""
        return sum(inv.estimated_cost_usd for inv in self.invocations)

    @computed_field
    @property
    def cache_hit_rate(self) -> float:
        """The percentage of invocations that were cache hits."""
        if not self.invocations:
            return 0.0
        hits = sum(1 for inv in self.invocations if inv.cache_hit)
        return hits / len(self.invocations)
