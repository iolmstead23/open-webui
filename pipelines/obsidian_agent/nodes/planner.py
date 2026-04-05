"""Planner node -- produces a PlanOutput from context and user intent.

Dependencies: llm, state, registry.
Owns: plan, retry_count fields.
"""
import json
import logging

from langchain_core.messages import SystemMessage, HumanMessage

from obsidian_agent.state.schema import AgentState, PlanOutput
from obsidian_agent.llm.client import invoke_structured
from obsidian_agent.registry.operations import OPERATIONS

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a planner for an Obsidian vault assistant. Your job is to produce a structured plan of Obsidian REST operations based on the user's request and the context provided.

## Available Operations
{operations_summary}

## Rules
1. Maximum 10 tool calls per plan.
2. NEVER plan DESTRUCTIVE operations unless the user's message explicitly contains the word "delete" or "remove".
3. NEVER write to file paths that were not discovered via a prior READ operation or explicitly named by the user.
4. Every operation MUST have a non-empty rationale explaining why it is needed.
5. Prefer obsidian_batch_get_file_contents over multiple obsidian_get_file_contents calls.
6. Prefer obsidian_patch_content over obsidian_append_content when the user specifies a location (heading, block, frontmatter field).
7. Paths are vault-relative, forward slashes, no leading slash. Paths are CASE-SENSITIVE.
8. For general_qa intent, return an empty calls list.

## Output Format
Think step by step about what operations are needed, then respond with a JSON object:
{{
  "calls": [
    {{
      "operation_name": "...",
      "inputs": {{}},
      "rationale": "...",
      "risk": "READ|WRITE|DESTRUCTIVE"
    }}
  ],
  "reasoning_trace": "Your step-by-step reasoning"
}}"""


def _build_operations_summary() -> str:
    """Build a concise summary of available operations for the planner prompt."""
    lines = []
    for name, op in OPERATIONS.items():
        req = ", ".join(f"{k}: {v}" for k, v in op["required_fields"].items())
        opt = ", ".join(f"{k} (default={v[1]})" for k, v in op.get("optional_fields", {}).items())
        fields = req
        if opt:
            fields += f" | optional: {opt}"
        lines.append(f"- **{name}** [{op['risk']}]: {op['description']} Fields: {fields or 'none'}")
    return "\n".join(lines)


def planner(state: AgentState, llm) -> dict:
    """Produce a PlanOutput from context and intent.

    On retry, injects plan_validation_errors into the prompt.

    Returns:
        Partial state dict with plan and incremented retry_count.
    """
    user_message = state.get("user_message", "")
    intent_data = state.get("intent") or {}
    intent = intent_data.get("intent", "general_qa") if isinstance(intent_data, dict) else "general_qa"
    aggregated_context = state.get("aggregated_context", "")
    validation_errors = state.get("plan_validation_errors", [])
    retry_count = state.get("retry_count", 0)

    system_prompt = SYSTEM_PROMPT.format(operations_summary=_build_operations_summary())

    messages = [SystemMessage(content=system_prompt)]

    # Build user prompt
    user_prompt_parts = [f"## User Request\n{user_message}"]
    user_prompt_parts.append(f"\n## Detected Intent: {intent}")

    if aggregated_context:
        user_prompt_parts.append(f"\n## Context\n{aggregated_context}")

    # On retry, include validation errors and previous plan
    if validation_errors and retry_count > 0:
        errors_text = "\n".join(f"- {e}" for e in validation_errors)
        user_prompt_parts.append(
            f"\n## VALIDATION ERRORS FROM PREVIOUS ATTEMPT (fix ALL of these)\n{errors_text}"
        )
        prev_plan = state.get("plan")
        if prev_plan:
            user_prompt_parts.append(
                f"\n## Previous rejected plan (for reference)\n```json\n{json.dumps(prev_plan, indent=2)}\n```"
            )

    messages.append(HumanMessage(content="\n".join(user_prompt_parts)))

    try:
        result = invoke_structured(llm, PlanOutput, messages)
        logger.info(
            "Planner produced %d calls (retry=%d)",
            len(result.calls),
            retry_count,
        )
        return {
            "plan": result.model_dump(),
            "retry_count": retry_count + 1,
        }
    except Exception as e:
        logger.error("Planner failed: %s", e)
        # Return empty plan that will fail validation, triggering retry or error
        empty = PlanOutput(calls=[], reasoning_trace=f"Planner error: {e}")
        return {
            "plan": empty.model_dump(),
            "retry_count": retry_count + 1,
            "error": f"[planner] {e}",
        }
