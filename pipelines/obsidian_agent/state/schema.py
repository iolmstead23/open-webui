"""State schema -- AgentState TypedDict and Pydantic sub-contracts.

Dependencies: stdlib, pydantic.
Exposed interface: AgentState, ObsidianToolCall, IntentOutput, PlanOutput.
"""
import operator
from typing import Any, Annotated, Literal, Optional
from typing_extensions import TypedDict
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Sub-contracts (Pydantic models for LLM structured output and validation)
# ---------------------------------------------------------------------------

VALID_INTENTS = Literal[
    "vault_read",
    "vault_write",
    "vault_delete",
    "general_qa",
    "periodic_note",
    "recent_changes",
]


class IntentOutput(BaseModel):
    """Output of the intent classifier node."""

    intent: VALID_INTENTS
    reasoning: str = Field(
        ..., description="Chain-of-thought explanation for the classification."
    )


class ObsidianToolCall(BaseModel):
    """A single planned Obsidian REST operation."""

    operation_name: str = Field(
        ..., description="Must match a key in the OPERATIONS registry."
    )
    inputs: dict[str, Any] = Field(
        ..., description="All required fields for the operation."
    )
    rationale: str = Field(
        ...,
        min_length=1,
        description="Planner justification for this call.",
    )
    risk: str = Field(
        ..., description="READ, WRITE, or DESTRUCTIVE -- copied from registry."
    )


class PlanOutput(BaseModel):
    """Output of the planner node."""

    calls: list[ObsidianToolCall] = Field(
        default_factory=list,
        description="Ordered sequence of operations. Empty list is valid for general_qa.",
    )
    reasoning_trace: str = Field(
        default="", description="Planner chain-of-thought."
    )


# ---------------------------------------------------------------------------
# AgentState TypedDict (parameterizes the LangGraph StateGraph)
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    """Shared state flowing through the LangGraph pipeline.

    Fields use default overwrite semantics unless annotated with a reducer.
    Only execution_results uses an append reducer (operator.add).
    """

    # Inlet (immutable after set)
    user_message: str
    conversation_history: list[dict]
    file_attachments: list[dict]
    image_attachments: list[dict]
    raw_search_results: list[dict]

    # Intent classification (write-once)
    intent: Optional[dict]  # serialized IntentOutput

    # Context gathering (write-once per owner)
    web_search_context: list[str]
    kb_results: list[dict]
    file_contents: list[dict]
    image_descriptions: list[str]

    # Aggregated context (write-once)
    aggregated_context: str

    # Planning (overwrite on retry)
    plan: Optional[dict]  # serialized PlanOutput
    plan_validation_errors: list[str]
    plan_valid: bool

    # Execution (append reducer)
    execution_results: Annotated[list[dict], operator.add]

    # Output (write-once)
    final_response: str

    # Error and retry tracking
    error: str
    retry_count: int
