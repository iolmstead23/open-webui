"""Obsidian API client -- HTTP wrapper for mcpo proxy at mcpo-1:8000.

Dependencies: config, registry.
Exposed interface: ObsidianClient class.
"""
import logging
from typing import Any

import httpx

from obsidian_agent.config import Settings
from obsidian_agent.registry import get_operation

logger = logging.getLogger(__name__)


class ObsidianClient:
    """Synchronous HTTP client for the mcpo Obsidian REST proxy.

    All requests are POST with JSON body to /obsidian/obsidian_<operation>.
    Authentication is handled by mcpo -- this client sends unauthenticated requests.
    """

    def __init__(self, settings: Settings) -> None:
        self.base_url = settings.mcpo_base_url
        self.client = httpx.Client(
            base_url=self.base_url,
            timeout=httpx.Timeout(
                connect=settings.mcpo_connect_timeout,
                read=settings.mcpo_read_timeout,
                write=settings.mcpo_read_timeout,
                pool=settings.mcpo_connect_timeout,
            ),
            headers={"Content-Type": "application/json"},
        )

    def call_tool(self, operation_name: str, inputs: dict[str, Any]) -> dict:
        """Call an mcpo-proxied Obsidian tool.

        Args:
            operation_name: Must match a key in the OPERATIONS registry.
            inputs: Dict of arguments for the operation.

        Returns:
            Dict with keys: success (bool), data (Any), status_code (int).

        Raises:
            httpx.HTTPStatusError: On non-2xx response (caller should handle).
        """
        op = get_operation(operation_name)
        if op is None:
            raise ValueError(f"Unknown operation: {operation_name}")

        endpoint = op["endpoint"]

        logger.info(
            "Calling %s at %s with inputs: %s",
            operation_name,
            endpoint,
            list(inputs.keys()),
        )

        response = self.client.post(endpoint, json=inputs)
        response.raise_for_status()

        # mcpo returns JSON for most operations
        try:
            data = response.json()
        except Exception:
            data = response.text

        logger.info(
            "Response from %s: status=%d",
            operation_name,
            response.status_code,
        )

        return {
            "success": True,
            "data": data,
            "status_code": response.status_code,
        }

    def health_check(self) -> bool:
        """Check connectivity to mcpo proxy."""
        try:
            response = self.client.get("/")
            return response.status_code < 500
        except Exception as e:
            logger.error("mcpo health check failed: %s", e)
            return False

    def close(self) -> None:
        """Close the HTTP client."""
        self.client.close()
