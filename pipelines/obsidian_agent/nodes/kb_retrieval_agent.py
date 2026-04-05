"""KB retrieval agent -- queries Obsidian vault for relevant notes.

Dependencies: obsidian, state.
Owns: kb_results field.
"""
import logging

from obsidian_agent.state.schema import AgentState
from obsidian_agent.obsidian.api_client import ObsidianClient

logger = logging.getLogger(__name__)


def kb_retrieval_agent(state: AgentState, obsidian_client: ObsidianClient) -> dict:
    """Query the Obsidian vault based on user intent.

    Routing logic:
    - vault_read / vault_write: simple search on user message keywords.
    - recent_changes: get recent changes.
    - periodic_note: get periodic note.
    - general_qa: skip (return empty).

    Returns:
        Partial state dict with kb_results.
    """
    intent_data = state.get("intent") or {}
    intent = intent_data.get("intent", "general_qa") if isinstance(intent_data, dict) else "general_qa"
    user_message = state.get("user_message", "")

    if intent == "general_qa":
        return {"kb_results": []}

    # Skip if user_message is an internal OpenWebUI task instruction
    if user_message.startswith("###"):
        return {"kb_results": []}

    try:
        if intent == "recent_changes":
            result = obsidian_client.call_tool("obsidian_get_recent_changes", {})
            return {"kb_results": [{"source": "recent_changes", "data": result["data"]}]}

        if intent == "periodic_note":
            # Detect period from message
            period = _detect_period(user_message)
            result = obsidian_client.call_tool(
                "obsidian_get_periodic_note", {"period": period}
            )
            return {"kb_results": [{"source": "periodic_note", "period": period, "data": result["data"]}]}

        # Default: simple search using keywords from user message
        search_query = _extract_search_terms(user_message)
        if search_query:
            result = obsidian_client.call_tool(
                "obsidian_simple_search",
                {"query": search_query, "context_length": 150},
            )
            return {"kb_results": [{"source": "simple_search", "query": search_query, "data": result["data"]}]}

        return {"kb_results": []}

    except Exception as e:
        logger.error("KB retrieval failed: %s", e)
        return {
            "kb_results": [],
            "error": f"[kb_retrieval_agent] {e}",
        }


def _detect_period(message: str) -> str:
    """Detect period type from user message."""
    msg = message.lower()
    if "daily" in msg or "today" in msg or "day" in msg:
        return "daily"
    if "weekly" in msg or "week" in msg:
        return "weekly"
    if "monthly" in msg or "month" in msg:
        return "monthly"
    if "quarterly" in msg or "quarter" in msg:
        return "quarterly"
    if "yearly" in msg or "year" in msg or "annual" in msg:
        return "yearly"
    return "daily"


def _extract_search_terms(message: str) -> str:
    """Extract meaningful search terms from a user message.

    Simple heuristic: strip common question words and return the core.
    """
    noise = {
        "what", "where", "when", "how", "who", "which", "can", "could",
        "would", "should", "do", "does", "did", "is", "are", "was", "were",
        "the", "a", "an", "my", "me", "i", "in", "on", "at", "to", "for",
        "of", "with", "about", "find", "search", "look", "get", "show",
        "list", "tell", "give", "have", "has", "notes", "note", "vault",
        "obsidian", "please", "thanks",
        # OpenWebUI internal task words
        "task", "analyze", "chat", "history", "context",
    }
    words = message.lower().split()
    terms = [w for w in words if w not in noise and len(w) > 2]
    return " ".join(terms[:5])
