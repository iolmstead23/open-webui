"""LLM client -- ChatOllama factory and structured output helper.

Dependencies: config.
Exposed interface: create_llm(), invoke_structured().
"""
import re
import json
import logging
from typing import Type, TypeVar

from langchain_ollama import ChatOllama
from pydantic import BaseModel

from obsidian_agent.config import Settings

logger = logging.getLogger(__name__)

ResponseModel = TypeVar("ResponseModel", bound=BaseModel)


def create_llm(settings: Settings, temperature: float = 0.0) -> ChatOllama:
    """Create a ChatOllama instance bound to the configured Ollama endpoint."""
    return ChatOllama(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
        temperature=temperature,
        num_predict=-1,
        timeout=settings.ollama_read_timeout,
    )


def invoke_structured(
    llm: ChatOllama,
    schema: Type[ResponseModel],
    messages: list,
) -> ResponseModel:
    """Invoke LLM and parse response into a Pydantic model.

    Uses a fallback-first approach:
    1. Try with_structured_output (Ollama constrained decoding).
    2. If that fails, invoke normally and parse JSON from the response,
       stripping <think> tags that deepseek-r1 produces.

    Args:
        llm: ChatOllama instance.
        schema: Pydantic BaseModel class to validate against.
        messages: List of langchain message objects.

    Returns:
        Validated instance of the schema.

    Raises:
        ValueError: If response cannot be parsed into the schema.
    """
    # Attempt 1: Ollama constrained JSON output
    try:
        structured_llm = llm.with_structured_output(schema, method="json_schema")
        result = structured_llm.invoke(messages)
        if result is not None:
            logger.info("Structured output succeeded for %s", schema.__name__)
            return result
    except Exception as e:
        logger.warning(
            "with_structured_output failed for %s: %s, falling back to manual parse",
            schema.__name__,
            e,
        )

    # Attempt 2: Manual parse with think-tag stripping
    response = llm.invoke(messages)
    content = response.content

    # Strip <think>...</think> blocks produced by deepseek-r1
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

    # Try to extract JSON from markdown code blocks
    json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", content)
    if json_match:
        raw_json = json_match.group(1).strip()
    else:
        # Try the whole content as JSON
        raw_json = content

    try:
        parsed = schema.model_validate_json(raw_json)
        logger.info("Manual JSON parse succeeded for %s", schema.__name__)
        return parsed
    except Exception as parse_err:
        # Last resort: try to find any JSON object in the content
        obj_match = re.search(r"\{[\s\S]*\}", content)
        if obj_match:
            try:
                parsed = schema.model_validate_json(obj_match.group(0))
                logger.info("Regex JSON extraction succeeded for %s", schema.__name__)
                return parsed
            except Exception:
                pass

        raise ValueError(
            f"Failed to parse LLM output into {schema.__name__}: {parse_err}. "
            f"Raw content: {content[:500]}"
        )
