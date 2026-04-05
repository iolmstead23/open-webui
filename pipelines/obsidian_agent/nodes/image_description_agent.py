"""Image description agent -- describes base64 image attachments via LLM.

Dependencies: llm, state.
Owns: image_descriptions field.
"""
import logging

from langchain_core.messages import HumanMessage

from obsidian_agent.state.schema import AgentState

logger = logging.getLogger(__name__)


def image_description_agent(state: AgentState, llm) -> dict:
    """Produce natural language descriptions of image attachments.

    Args:
        state: Current pipeline state.
        llm: ChatOllama instance.

    Returns:
        Partial state dict with image_descriptions.
    """
    images = state.get("image_attachments", [])

    if not images:
        return {"image_descriptions": []}

    try:
        descriptions = []
        for image in images:
            url = image.get("url", image.get("data", ""))
            if not url:
                continue

            message = HumanMessage(
                content=[
                    {"type": "text", "text": "Describe this image concisely in 1-2 sentences."},
                    {"type": "image_url", "image_url": {"url": url}},
                ]
            )

            try:
                response = llm.invoke([message])
                descriptions.append(response.content)
            except Exception as e:
                logger.warning("Failed to describe image: %s", e)
                descriptions.append("[Image description unavailable]")

        logger.info("Described %d images", len(descriptions))
        return {"image_descriptions": descriptions}

    except Exception as e:
        logger.error("Image description failed: %s", e)
        return {
            "image_descriptions": [],
            "error": f"[image_description_agent] {e}",
        }
