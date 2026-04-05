"""
title: Obsidian Agent Pipeline
author: Third Eye Consulting
version: 1.0.0
requirements: langgraph>=0.2.0, langchain-ollama>=0.2.0, langchain-core>=0.3.0, pydantic>=2.0, httpx>=0.27
"""
import sys
import os
import logging
from typing import List, Optional, Union, Generator, Iterator

from pydantic import BaseModel, Field

logger = logging.getLogger("obsidian_agent_pipeline")

# Inject pipelines directory into sys.path so obsidian_agent package is importable.
# Must happen at module level before any obsidian_agent imports.
_pipelines_dir = os.path.dirname(os.path.abspath(__file__))
if _pipelines_dir not in sys.path:
    sys.path.insert(0, _pipelines_dir)


class Pipeline:
    """OpenWebUI Pipeline for Obsidian vault operations via LangGraph."""

    class Valves(BaseModel):
        pipelines: List[str] = ["*"]
        priority: int = 0
        MCPO_BASE_URL: str = Field(
            default="",
            description="mcpo proxy base URL (e.g. http://mcpo-1:8000)",
        )
        OLLAMA_BASE_URL: str = Field(
            default="",
            description="Ollama host URL (e.g. http://host.docker.internal:11434)",
        )
        OLLAMA_MODEL: str = Field(
            default="",
            description="Ollama model ID (e.g. deepseek-r1:latest)",
        )
        MAX_PLAN_RETRIES: int = Field(
            default=3,
            description="Planner retry ceiling before error",
        )
        CONTEXT_TOKEN_BUDGET: int = Field(
            default=6000,
            description="Token budget for context aggregation",
        )
        OLLAMA_READ_TIMEOUT: int = Field(
            default=120,
            description="HTTP read timeout in seconds for LLM calls",
        )

    def __init__(self):
        self.name = "Obsidian Agent Pipeline"
        self.valves = self.Valves(
            **{
                "MCPO_BASE_URL": os.getenv("MCPO_BASE_URL", ""),
                "OLLAMA_BASE_URL": os.getenv("OLLAMA_BASE_URL", ""),
                "OLLAMA_MODEL": os.getenv("OLLAMA_MODEL", ""),
                "MAX_PLAN_RETRIES": int(os.getenv("MAX_PLAN_RETRIES", "3")),
                "CONTEXT_TOKEN_BUDGET": int(os.getenv("CONTEXT_TOKEN_BUDGET", "6000")),
                "OLLAMA_READ_TIMEOUT": int(os.getenv("OLLAMA_READ_TIMEOUT", "120")),
            }
        )
        self.graph = None
        self.obsidian_client = None

    async def on_startup(self):
        """Build config, compile graph, validate connectivity."""
        # Lazy imports: obsidian_agent sub-packages are imported here (not at module
        # level) so the Pipeline class is always visible to OpenWebUI's loader even
        # before requirements are installed.
        from obsidian_agent.config import get_settings
        from obsidian_agent.graph.builder import build_graph
        from obsidian_agent.obsidian.api_client import ObsidianClient

        logger.info("Obsidian Agent Pipeline starting up...")

        try:
            env = {
                "MCPO_BASE_URL": self.valves.MCPO_BASE_URL,
                "OLLAMA_BASE_URL": self.valves.OLLAMA_BASE_URL,
                "OLLAMA_MODEL": self.valves.OLLAMA_MODEL,
                "MAX_PLAN_RETRIES": str(self.valves.MAX_PLAN_RETRIES),
                "CONTEXT_TOKEN_BUDGET": str(self.valves.CONTEXT_TOKEN_BUDGET),
                "OLLAMA_READ_TIMEOUT": str(self.valves.OLLAMA_READ_TIMEOUT),
            }
            settings = get_settings(env)

            self.obsidian_client = ObsidianClient(settings)
            self.graph = build_graph(settings, self.obsidian_client)

            if self.obsidian_client.health_check():
                logger.info("mcpo proxy connectivity: OK")
            else:
                logger.warning("mcpo proxy connectivity: FAILED (pipeline may not work)")

            logger.info("Obsidian Agent Pipeline started successfully")

        except Exception as e:
            logger.error("Obsidian Agent Pipeline startup failed: %s", e, exc_info=True)
            raise

    async def on_shutdown(self):
        """Clean up resources."""
        if self.obsidian_client:
            self.obsidian_client.close()
        logger.info("Obsidian Agent Pipeline shut down")

    async def inlet(self, body: dict, user: Optional[dict] = None) -> dict:
        """Normalize incoming request and extract attachments."""
        messages = body.get("messages", [])
        if not messages:
            return body

        last_message = messages[-1]
        content = last_message.get("content", "")

        file_attachments = []
        image_attachments = []

        if isinstance(content, list):
            text_parts = []
            for item in content:
                if not isinstance(item, dict):
                    continue
                item_type = item.get("type", "")
                if item_type == "text":
                    text_parts.append(item.get("text", ""))
                elif item_type == "image_url":
                    url = item.get("image_url", {}).get("url", "")
                    if url:
                        image_attachments.append({"url": url})
                else:
                    file_attachments.append(item)
            body["_extracted_text"] = " ".join(text_parts)

        body["file_attachments"] = file_attachments
        body["image_attachments"] = image_attachments
        body["raw_search_results"] = body.get("web_search_results", [])
        return body

    def pipe(
        self,
        user_message: str,
        model_id: str,
        messages: List[dict],
        body: dict,
    ) -> Union[str, Generator, Iterator]:
        """Invoke the LangGraph graph. Always returns a string."""
        try:
            if self.graph is None:
                return "Pipeline not initialized. Check startup logs for errors."

            actual_message = body.get("_extracted_text", user_message)

            initial_state = {
                "user_message": actual_message,
                "conversation_history": messages[:-1] if len(messages) > 1 else [],
                "file_attachments": body.get("file_attachments", []),
                "image_attachments": body.get("image_attachments", []),
                "raw_search_results": body.get("raw_search_results", []),
                "intent": None,
                "web_search_context": [],
                "kb_results": [],
                "file_contents": [],
                "image_descriptions": [],
                "aggregated_context": "",
                "plan": None,
                "plan_validation_errors": [],
                "plan_valid": False,
                "execution_results": [],
                "final_response": "",
                "error": "",
                "retry_count": 0,
            }

            result = self.graph.invoke(initial_state)

            response = result.get("final_response", "")
            if not response:
                response = "No response generated. Please try again."
            return response

        except Exception as e:
            logger.error("Pipeline error: %s", e, exc_info=True)
            return (
                "An error occurred while processing your request. "
                "Please try rephrasing or simplifying."
            )

    async def outlet(self, body: dict, user: Optional[dict] = None) -> dict:
        """Replace non-ASCII characters in the last message content with '?'.

        Open Web UI may reject responses containing high-byte characters depending
        on the model. This guard encodes to ASCII with the 'replace' error handler,
        substituting any non-ASCII code point with a literal '?'.
        """
        messages = body.get("messages", [])
        if messages:
            last = messages[-1]
            content = last.get("content", "")
            if isinstance(content, str):
                last["content"] = content.encode("ascii", errors="replace").decode("ascii")
        return body
