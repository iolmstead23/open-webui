"""Web search extractor node -- normalizes SearchAPI results from OpenWebUI.

Dependencies: state.
Owns: web_search_context field.
"""
import logging

from obsidian_agent.state.schema import AgentState

logger = logging.getLogger(__name__)


def web_search_extractor(state: AgentState) -> dict:
    """Extract and normalize SearchAPI results injected by OpenWebUI.

    Reads raw_search_results (array of objects with title, url, snippet)
    and normalizes each to a plain string for the context aggregator.

    Returns:
        Partial state dict with web_search_context.
    """
    raw_results = state.get("raw_search_results", [])

    if not raw_results:
        return {"web_search_context": []}

    try:
        context = []
        for item in raw_results:
            if isinstance(item, dict):
                title = item.get("title", "")
                url = item.get("url", "")
                snippet = item.get("snippet", item.get("content", ""))
                context.append(f"[{title}]({url}): {snippet}")
            elif isinstance(item, str):
                context.append(item)

        logger.info("Extracted %d web search results", len(context))
        return {"web_search_context": context}

    except Exception as e:
        logger.error("Web search extraction failed: %s", e)
        return {
            "web_search_context": [],
            "error": f"[web_search_extractor] {e}",
        }
