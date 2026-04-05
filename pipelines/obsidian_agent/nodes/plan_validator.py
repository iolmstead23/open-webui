"""Plan validator node -- validates plan against typed rules.

Dependencies: registry, state.
Owns: plan_valid, plan_validation_errors fields.
"""
import logging
import re

from obsidian_agent.state.schema import AgentState
from obsidian_agent.registry.operations import OPERATIONS

logger = logging.getLogger(__name__)


def plan_validator(state: AgentState) -> dict:
    """Apply all validation rules to the plan.

    Rules (each falsifiable, one error per failed rule):
    1. plan.calls length <= 10
    2. Every operation_name exists in OPERATIONS registry
    3. Every required input field present
    4. No DESTRUCTIVE operation unless "delete"/"remove" in user_message
    5. DESTRUCTIVE filepath must end in .md unless folder deletion intended
    6. Every call has non-empty rationale
    7. obsidian_delete_file calls must have confirm=true
    8. Empty calls list is valid
    9. (Security) No path traversal in filepaths

    Returns:
        Partial state dict with plan_valid and plan_validation_errors.
    """
    plan = state.get("plan") or {}
    calls = plan.get("calls", [])
    user_message = state.get("user_message", "")
    user_lower = user_message.lower()

    errors: list[str] = []

    # Rule 8: empty calls list is valid
    if not calls:
        logger.info("Plan validator: empty plan, valid")
        return {"plan_valid": True, "plan_validation_errors": []}

    # Rule 1: max 10 calls
    if len(calls) > 10:
        errors.append(f"Plan exceeds maximum of 10 calls (has {len(calls)})")

    for i, call in enumerate(calls):
        op_name = call.get("operation_name", "")
        inputs = call.get("inputs", {})
        rationale = call.get("rationale", "")
        risk = call.get("risk", "")

        # Rule 2: operation must exist
        op_entry = OPERATIONS.get(op_name)
        if op_entry is None:
            errors.append(f"Call {i}: unknown operation '{op_name}'")
            continue

        # Rule 3: required fields present
        for field_name in op_entry["required_fields"]:
            if field_name not in inputs:
                errors.append(
                    f"Call {i} ({op_name}): missing required field: {field_name}"
                )

        # Rule 4: DESTRUCTIVE requires delete/remove in user message
        if op_entry["risk"] == "DESTRUCTIVE":
            if "delete" not in user_lower and "remove" not in user_lower:
                errors.append(
                    f"Call {i} ({op_name}): destructive operation requires "
                    f"'delete' or 'remove' in user message"
                )

        # Rule 5: DESTRUCTIVE filepath must end in .md
        if op_entry["risk"] == "DESTRUCTIVE":
            filepath = inputs.get("filepath", "")
            if filepath and not filepath.endswith(".md"):
                # Allow folder deletion only if user explicitly mentions it
                if "folder" not in user_lower and "directory" not in user_lower:
                    errors.append(
                        f"Call {i} ({op_name}): filepath '{filepath}' must end "
                        f"in .md for delete operations unless folder deletion "
                        f"is explicitly intended"
                    )

        # Rule 6: non-empty rationale
        if not rationale or not rationale.strip():
            errors.append(f"Call {i} ({op_name}): empty rationale")

        # Rule 7: obsidian_delete_file must have confirm=true
        if op_name == "obsidian_delete_file":
            if inputs.get("confirm") is not True:
                errors.append(
                    f"Call {i} ({op_name}): confirm must be true"
                )

        # Rule 9 (security): path traversal prevention
        for field_name in ("filepath", "dirpath"):
            path_val = inputs.get(field_name, "")
            if path_val and not _is_safe_vault_path(path_val):
                errors.append(
                    f"Call {i} ({op_name}): invalid path: traversal detected "
                    f"in '{field_name}' value '{path_val}'"
                )

    plan_valid = len(errors) == 0

    if plan_valid:
        logger.info("Plan validated: %d calls, all rules passed", len(calls))
    else:
        logger.warning("Plan validation failed with %d errors: %s", len(errors), errors)

    return {
        "plan_valid": plan_valid,
        "plan_validation_errors": errors,
    }


def _is_safe_vault_path(path: str) -> bool:
    """Check that a path is safe (no traversal, no absolute paths)."""
    if ".." in path:
        return False
    if path.startswith("/") or path.startswith("\\"):
        return False
    if ":" in path:
        return False
    return True
