"""Context aggregator node -- merges all context sources with token budget.

Dependencies: config, state.
Owns: aggregated_context field.
"""
import json
import logging

from obsidian_agent.state.schema import AgentState

logger = logging.getLogger(__name__)


def context_aggregator(state: AgentState, token_budget: int, chars_per_token: float) -> dict:
    """Merge all context sources into a single token-budget-limited string.

    Truncation priority (highest retention to lowest):
    1. conversation_history (min 2 recent turns)
    2. kb_results (min 1 result)
    3. file_contents
    4. web_search_context
    5. image_descriptions

    Args:
        state: Current pipeline state.
        token_budget: Max tokens for aggregated context.
        chars_per_token: Character-to-token ratio for estimation.

    Returns:
        Partial state dict with aggregated_context.
    """
    char_budget = int(token_budget * chars_per_token)

    sections = []
    remaining = char_budget

    # 1. Conversation history (min 2 recent turns)
    history = state.get("conversation_history", [])
    if history:
        recent = history[-4:]  # last 2 exchanges (4 messages)
        hist_text = _format_history(recent)
        hist_chars = min(len(hist_text), remaining)
        sections.append(f"### Conversation Context\n{hist_text[:hist_chars]}")
        remaining -= hist_chars

    # 2. KB results (min 1 result)
    kb_results = state.get("kb_results", [])
    if kb_results and remaining > 200:
        kb_text = _format_kb_results(kb_results)
        kb_chars = min(len(kb_text), remaining)
        sections.append(f"### Vault Search Results\n{kb_text[:kb_chars]}")
        remaining -= kb_chars

    # 3. File contents
    file_contents = state.get("file_contents", [])
    if file_contents and remaining > 200:
        fc_text = _format_file_contents(file_contents)
        fc_chars = min(len(fc_text), remaining)
        sections.append(f"### File Attachments\n{fc_text[:fc_chars]}")
        remaining -= fc_chars

    # 4. Web search context
    web_context = state.get("web_search_context", [])
    if web_context and remaining > 100:
        web_text = "\n".join(web_context)
        web_chars = min(len(web_text), remaining)
        sections.append(f"### Web Search Results\n{web_text[:web_chars]}")
        remaining -= web_chars

    # 5. Image descriptions
    img_desc = state.get("image_descriptions", [])
    if img_desc and remaining > 50:
        img_text = "\n".join(img_desc)
        img_chars = min(len(img_text), remaining)
        sections.append(f"### Image Descriptions\n{img_text[:img_chars]}")

    aggregated = "\n\n".join(sections)
    logger.info(
        "Aggregated context: %d chars (%d estimated tokens, budget %d)",
        len(aggregated),
        int(len(aggregated) / chars_per_token),
        token_budget,
    )

    return {"aggregated_context": aggregated}


def _format_history(turns: list[dict]) -> str:
    """Render a list of conversation turn dictionaries into labeled Markdown lines.

    Each turn is expected to have 'role' and 'content' keys. Multimodal
    content (list of content parts) is flattened to its text parts only.

    Returns:
        Newline-joined string with each line formatted as '**role**: content'.
    """
    lines = []
    for turn in turns:
        role = turn.get("role", "unknown")
        content = turn.get("content", "")
        if isinstance(content, list):
            # Multimodal content -- extract text parts
            content = " ".join(
                item.get("text", "") for item in content if isinstance(item, dict) and item.get("type") == "text"
            )
        lines.append(f"**{role}**: {content}")
    return "\n".join(lines)


def _format_kb_results(results: list[dict]) -> str:
    """Render a list of vault knowledge-base result dictionaries into a flat Markdown string.

    Each result dictionary has 'source' and 'data' keys. When data is a list of file
    match objects, up to 10 files and 3 context snippets per file are rendered.
    String data is included verbatim (truncated to 2000 chars); other types are
    JSON-serialized (truncated to 2000 chars).

    Returns:
        Newline-joined string of formatted knowledge-base result lines.
    """
    lines = []
    for kb_result in results:
        source = kb_result.get("source", "unknown")
        data = kb_result.get("data", "")
        if isinstance(data, list):
            for item in data[:10]:
                if isinstance(item, dict):
                    filename = item.get("filename", item.get("path", ""))
                    lines.append(f"- `{filename}`")
                    matches = item.get("matches", [])
                    for match in matches[:3]:
                        ctx = match.get("context", "")
                        if ctx:
                            lines.append(f"  > {ctx[:200]}")
                else:
                    lines.append(f"- {str(item)[:200]}")
        elif isinstance(data, str):
            lines.append(data[:2000])
        else:
            lines.append(json.dumps(data, indent=2)[:2000])
    return "\n".join(lines)


def _format_file_contents(contents: list[dict]) -> str:
    """Render a list of extracted file-content dictionaries into a Markdown string.

    Each dictionary has 'filename' and 'content' keys. Content is truncated to
    1000 characters per file. Files are separated by blank lines.

    Returns:
        Double-newline-joined string with each file rendered as a bold filename header followed by its content on the next line.
    """
    lines = []
    for file_entry in contents:
        name = file_entry.get("filename", "unknown")
        content = file_entry.get("content", "")
        lines.append(f"**{name}**:\n{content[:1000]}")
    return "\n\n".join(lines)
