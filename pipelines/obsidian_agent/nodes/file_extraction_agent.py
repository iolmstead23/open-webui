"""File extraction agent -- extracts text from non-image file attachments.

Dependencies: state.
Owns: file_contents field.
"""
import base64
import logging

from obsidian_agent.state.schema import AgentState

logger = logging.getLogger(__name__)


def file_extraction_agent(state: AgentState) -> dict:
    """Extract text content from file attachments.

    Processes attachment data already in state (no external calls).

    Returns:
        Partial state dict with file_contents.
    """
    attachments = state.get("file_attachments", [])

    if not attachments:
        return {"file_contents": []}

    try:
        contents = []
        for attachment in attachments:
            name = attachment.get("name", attachment.get("filename", "unknown"))
            data = attachment.get("data", attachment.get("content", ""))

            # Try base64 decode if it looks encoded
            if isinstance(data, str) and data.startswith("data:"):
                # Strip data URI prefix
                _, _, encoded = data.partition(",")
                try:
                    data = base64.b64decode(encoded).decode("utf-8", errors="replace")
                except Exception:
                    pass

            contents.append({"filename": name, "content": str(data)[:5000]})

        logger.info("Extracted content from %d file attachments", len(contents))
        return {"file_contents": contents}

    except Exception as e:
        logger.error("File extraction failed: %s", e)
        return {
            "file_contents": [],
            "error": f"[file_extraction_agent] {e}",
        }
