"""Configuration settings -- reads environment variables, exposes immutable Settings.

Dependencies: stdlib only.
Exposed interface: Settings dataclass, get_settings() factory.
"""
import os
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

REQUIRED_VARS = [
    "MCPO_BASE_URL",
    "OLLAMA_BASE_URL",
    "OLLAMA_MODEL",
]

OPTIONAL_INT_VARS = {
    "MAX_PLAN_RETRIES": 3,
    "CONTEXT_TOKEN_BUDGET": 6000,
    "OLLAMA_READ_TIMEOUT": 120,
}


@dataclass(frozen=True)
class Settings:
    """Immutable runtime configuration for the pipeline."""

    mcpo_base_url: str
    ollama_base_url: str
    ollama_model: str
    max_plan_retries: int
    context_token_budget: int
    ollama_read_timeout: int

    # Derived constants
    mcpo_connect_timeout: float = 5.0
    mcpo_read_timeout: float = 30.0
    chars_per_token: float = 3.5


def get_settings(env: dict[str, str] | None = None) -> Settings:
    """Build Settings from environment variables.

    Args:
        env: Optional dict to read from instead of os.environ.
             Useful for testing.

    Raises:
        ValueError: If a required environment variable is missing.
    """
    source = env if env is not None else dict(os.environ)

    missing = [v for v in REQUIRED_VARS if v not in source or not source[v]]
    if missing:
        raise ValueError(
            f"Missing required environment variable(s): {', '.join(missing)}"
        )

    int_values = {}
    for var, default in OPTIONAL_INT_VARS.items():
        raw = source.get(var, "")
        if raw:
            try:
                int_values[var] = int(raw)
            except ValueError:
                raise ValueError(
                    f"Environment variable {var} must be an integer, got: {raw!r}"
                )
        else:
            int_values[var] = default

    settings = Settings(
        mcpo_base_url=source["MCPO_BASE_URL"].rstrip("/"),
        ollama_base_url=source["OLLAMA_BASE_URL"].rstrip("/"),
        ollama_model=source["OLLAMA_MODEL"],
        max_plan_retries=int_values["MAX_PLAN_RETRIES"],
        context_token_budget=int_values["CONTEXT_TOKEN_BUDGET"],
        ollama_read_timeout=int_values["OLLAMA_READ_TIMEOUT"],
    )

    logger.info(
        "Settings loaded: mcpo=%s ollama=%s model=%s retries=%d budget=%d timeout=%d",
        settings.mcpo_base_url,
        settings.ollama_base_url,
        settings.ollama_model,
        settings.max_plan_retries,
        settings.context_token_budget,
        settings.ollama_read_timeout,
    )

    return settings
