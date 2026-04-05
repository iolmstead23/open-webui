"""Executor node -- executes planned ObsidianToolCalls against mcpo proxy.

Dependencies: obsidian, state, registry.
Owns: execution_results field (append reducer).
"""
import logging

import httpx

from obsidian_agent.state.schema import AgentState
from obsidian_agent.obsidian.api_client import ObsidianClient

logger = logging.getLogger(__name__)


def executor(state: AgentState, obsidian_client: ObsidianClient) -> dict:
    """Execute each planned tool call sequentially against mcpo-1:8000.

    Execution contract:
    - Fire calls in list order.
    - Each response appended to execution_results.
    - READ failure: record error, continue to next call.
    - WRITE/DESTRUCTIVE failure: halt immediately, write to error.

    Returns:
        Partial state dict with execution_results (uses append reducer).
    """
    plan = state.get("plan") or {}
    calls = plan.get("calls", [])

    if not calls:
        return {"execution_results": []}

    results = []

    for call_index, call in enumerate(calls):
        op_name = call.get("operation_name", "")
        inputs = call.get("inputs", {})
        risk = call.get("risk", "READ")

        logger.info(
            "Executing call %d/%d: %s (%s)",
            call_index + 1,
            len(calls),
            op_name,
            risk,
        )

        try:
            response = obsidian_client.call_tool(op_name, inputs)
            results.append({
                "operation_name": op_name,
                "inputs": inputs,
                "response": response.get("data"),
                "success": True,
            })
            logger.info("Call %d succeeded: %s", call_index + 1, op_name)

        except httpx.HTTPStatusError as e:
            error_detail = f"HTTP {e.response.status_code}: {e.response.text[:200]}"
            logger.error("Call %d failed: %s - %s", call_index + 1, op_name, error_detail)

            results.append({
                "operation_name": op_name,
                "inputs": inputs,
                "response": None,
                "success": False,
                "error": error_detail,
            })

            # Halt on WRITE/DESTRUCTIVE failure
            if risk in ("WRITE", "DESTRUCTIVE"):
                return {
                    "execution_results": results,
                    "error": f"[executor] {op_name} failed ({risk}): {error_detail}",
                }

        except Exception as e:
            error_detail = f"{type(e).__name__}: {e}"
            logger.error("Call %d failed: %s - %s", call_index + 1, op_name, error_detail)

            results.append({
                "operation_name": op_name,
                "inputs": inputs,
                "response": None,
                "success": False,
                "error": error_detail,
            })

            if risk in ("WRITE", "DESTRUCTIVE"):
                return {
                    "execution_results": results,
                    "error": f"[executor] {op_name} failed ({risk}): {error_detail}",
                }

    return {"execution_results": results}
