"""Summarizer node -- produces final natural language response.

Dependencies: llm, state.
Owns: final_response field.
"""
import json
import logging

from langchain_core.messages import SystemMessage, HumanMessage

from obsidian_agent.state.schema import AgentState

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a helpful Obsidian vault assistant. Produce a clear, concise response for the user based on the execution results and context provided.

## Output Rules
- Include every file path touched and every operation performed with its outcome.
- Summarize relevant vault content found -- do NOT dump large blocks of raw vault text.
- Use ASCII-only markdown. Use code fences for file paths and note content excerpts.
- No headers above H3 (###).
- Do not mention internal state fields, retry history, or validation errors.
- If this is a general_qa response (no vault operations), respond using only the context and user message."""


def summarizer(state: AgentState, llm) -> dict:
    """Produce the final response from execution results and context.

    Returns:
        Partial state dict with final_response.
    """
    user_message = state.get("user_message", "")
    intent_data = state.get("intent") or {}
    intent = intent_data.get("intent", "general_qa") if isinstance(intent_data, dict) else "general_qa"
    execution_results = state.get("execution_results", [])
    aggregated_context = state.get("aggregated_context", "")

    messages = [SystemMessage(content=SYSTEM_PROMPT)]

    # Build content for the summarizer
    parts = [f"## User Request\n{user_message}"]

    if aggregated_context:
        parts.append(f"\n## Context\n{aggregated_context}")

    if execution_results:
        results_text = _format_execution_results(execution_results)
        parts.append(f"\n## Execution Results\n{results_text}")
    elif intent != "general_qa":
        parts.append("\n## Execution Results\nNo vault operations were executed.")

    messages.append(HumanMessage(content="\n".join(parts)))

    try:
        response = llm.invoke(messages)
        content = response.content

        # Strip any <think> tags from deepseek-r1
        import re
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

        logger.info("Summarizer produced %d char response", len(content))
        return {"final_response": content}

    except Exception as e:
        logger.error("Summarizer failed: %s", e)
        # Produce a basic response without LLM
        fallback = _build_fallback_response(user_message, execution_results)
        return {"final_response": fallback}


def _format_execution_results(results: list[dict]) -> str:
    """Render a list of executor result dictionaries into a Markdown bullet list.

    Each dictionary is expected to have 'operation_name', 'success', 'inputs',
    and optionally 'response' and 'error' keys. Successful responses are
    truncated to 500 chars; list responses report only the item count.

    Returns:
        Newline-joined Markdown string, one bullet per operation.
    """
    lines = []
    for result in results:
        op = result.get("operation_name", "unknown")
        success = result.get("success", False)
        status = "OK" if success else "FAILED"
        inputs = result.get("inputs", {})

        filepath = inputs.get("filepath", inputs.get("dirpath", inputs.get("query", "")))
        lines.append(f"- **{op}** [{status}] target: `{filepath}`")

        if success and result.get("response"):
            resp = result["response"]
            if isinstance(resp, str):
                lines.append(f"  Response: {resp[:500]}")
            elif isinstance(resp, list):
                if not resp:
                    lines.append(f"  No results found")
                else:
                    lines.append(f"  Returned {len(resp)} results:")
                    for item in resp[:5]:
                        if isinstance(item, dict):
                            fname = item.get("filename", item.get("path", "?"))
                            lines.append(f"    - `{fname}`")
                            for m in item.get("matches", [])[:2]:
                                ctx = m.get("context", "")
                                if ctx:
                                    lines.append(f"      > {ctx[:200]}")
                        else:
                            lines.append(f"    - {str(item)[:200]}")
            elif isinstance(resp, dict):
                lines.append(f"  Response: {json.dumps(resp, indent=2)[:500]}")

        if not success and result.get("error"):
            lines.append(f"  Error: {result['error']}")

    return "\n".join(lines)


def _build_fallback_response(user_message: str, results: list[dict]) -> str:
    """Build a basic response without LLM when summarizer fails."""
    lines = ["I processed your request but encountered an issue generating a summary.\n"]

    if results:
        lines.append("### Operations performed:")
        for r in results:
            op = r.get("operation_name", "unknown")
            success = "succeeded" if r.get("success") else "failed"
            lines.append(f"- `{op}`: {success}")

    return "\n".join(lines)
