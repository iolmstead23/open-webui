"""Error handler node -- produces user-facing error response.

Dependencies: state.
Owns: final_response field (on error paths only).
"""
import logging

from obsidian_agent.state.schema import AgentState

logger = logging.getLogger(__name__)

FALLBACK_MESSAGE = (
    "An unexpected error occurred while processing your request. "
    "Please try rephrasing or simplifying your request."
)


def error_handler(state: AgentState) -> dict:
    """Produce a user-facing error response.

    Output includes:
    - What the user was attempting.
    - The nature of the failure.
    - Whether retries were exhausted.
    - Suggestion to rephrase.

    Must NOT expose raw stack traces or internal state field names.

    Returns:
        Partial state dict with final_response.
    """
    error = state.get("error", "")
    user_message = state.get("user_message", "")
    retry_count = state.get("retry_count", 0)
    plan_errors = state.get("plan_validation_errors", [])

    try:
        parts = []

        # What the user was trying to do
        if user_message:
            parts.append(
                f"I was unable to complete your request: \"{user_message[:200]}\""
            )

        # Nature of the failure
        if plan_errors:
            parts.append(
                f"\nThe operation plan could not be validated after {retry_count} "
                f"attempt(s). The planner was unable to produce a valid set of "
                f"operations for your request."
            )
        elif error:
            # Sanitize error -- remove anything that looks like internal state
            sanitized = _sanitize_error(error)
            parts.append(f"\nThe operation failed: {sanitized}")
        else:
            parts.append("\nAn unexpected error occurred during processing.")

        # Suggestion
        parts.append(
            "\nYou may try:\n"
            "- Rephrasing your request more specifically\n"
            "- Breaking a complex request into simpler steps\n"
            "- Checking that the files or folders you referenced exist in your vault"
        )

        response = "\n".join(parts)
        logger.info("Error handler produced response (%d chars)", len(response))
        return {"final_response": response}

    except Exception as e:
        logger.error("Error handler itself failed: %s", e)
        return {"final_response": FALLBACK_MESSAGE}


def _sanitize_error(error: str) -> str:
    """Remove internal details from error messages."""
    # Remove bracketed node names like [executor]
    import re
    sanitized = re.sub(r"\[[\w_]+\]\s*", "", error)
    # Truncate long errors
    if len(sanitized) > 300:
        sanitized = sanitized[:300] + "..."
    return sanitized
