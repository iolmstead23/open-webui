"""
title: Obsidian Agent Pipeline
author: Third Eye Consulting
version: 1.0.0
requirements: langgraph==1.1.6, langchain-ollama==1.0.1, langchain-core==1.2.26, pydantic==2.12.5, httpx==0.28.1, tenacity==9.1.4
"""

# ── Safe top-level imports (pre-installed in OpenWebUI container) ──────────────
# langchain_ollama, langchain_core, langgraph are NOT imported here.
# OpenWebUI scans this file BEFORE installing requirements to read the
# requirements: docstring above. Any top-level import of a not-yet-installed
# package crashes the module load and hides the Pipeline class from the loader.
# All heavy imports are deferred to the functions that need them.
import base64, json, logging, operator, os, re, sys
from dataclasses import dataclass
from functools import partial
from typing import (
    Any,
    Annotated,
    Generator,
    Iterator,
    List,
    Literal,
    Optional,
    Type,
    TypeVar,
    Union,
)
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception

logger = logging.getLogger("obsidian_agent_pipeline")
# ── state/schema ──
VALID_INTENTS = Literal[
    "vault_read",
    "vault_write",
    "vault_delete",
    "general_qa",
    "periodic_note",
    "recent_changes",
]


class IntentOutput(BaseModel):
    intent: VALID_INTENTS
    reasoning: str = Field(
        ..., description="Chain-of-thought explanation for the classification."
    )


class ObsidianToolCall(BaseModel):
    operation_name: str = Field(
        ..., description="Must match a key in the OPERATIONS registry."
    )
    inputs: dict[str, Any] = Field(
        ..., description="All required fields for the operation."
    )
    rationale: str = Field(
        ..., min_length=1, description="Planner justification for this call."
    )
    risk: str = Field(
        ..., description="READ, WRITE, or DESTRUCTIVE -- copied from registry."
    )


class PlanOutput(BaseModel):
    calls: list[ObsidianToolCall] = Field(
        default_factory=list, description="Ordered sequence of operations."
    )
    reasoning_trace: str = Field(default="", description="Planner chain-of-thought.")


class ExecutionResult(TypedDict, total=False):
    operation_name: str
    inputs: dict
    response: Any
    success: bool
    error: str


def make_node_error(node_name: str, exc: Exception) -> str:
    return f"[{node_name}] {exc}"


def get_intent(intent_data: Any, default: str = "general_qa") -> str:
    if isinstance(intent_data, dict):
        return intent_data.get("intent", default)
    return default


class AgentState(TypedDict):
    user_message: str
    conversation_history: list[dict]
    file_attachments: list[dict]
    image_attachments: list[dict]
    raw_search_results: list[dict]
    intent: Optional[dict]
    web_search_context: list[str]
    kb_results: list[dict]
    file_contents: list[dict]
    image_descriptions: list[str]
    aggregated_context: str
    plan: Optional[dict]
    plan_validation_errors: list[str]
    plan_valid: bool
    execution_results: Annotated[list[dict], operator.add]
    final_response: str
    error: str
    retry_count: int


# ── config/settings ──
REQUIRED_VARS = ["MCPO_BASE_URL", "OLLAMA_BASE_URL", "OLLAMA_MODEL"]
OPTIONAL_INT_VARS = {
    "MAX_PLAN_RETRIES": 3,
    "CONTEXT_TOKEN_BUDGET": 4000,
    "OLLAMA_READ_TIMEOUT": 120,
}
OPTIONAL_FLOAT_VARS = {"MCPO_CONNECT_TIMEOUT": 5.0, "MCPO_READ_TIMEOUT": 30.0}
CONVERSATION_WINDOW_TURNS: int = 4


@dataclass(frozen=True)
class Settings:
    mcpo_base_url: str
    ollama_base_url: str
    ollama_model: str
    max_plan_retries: int
    context_token_budget: int
    ollama_read_timeout: int
    mcpo_connect_timeout: float
    mcpo_read_timeout: float
    chars_per_token: float = 3.5


def get_settings(env: dict[str, str] | None = None) -> Settings:
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
    float_values = {}
    for var, default in OPTIONAL_FLOAT_VARS.items():
        raw = source.get(var, "")
        if raw:
            try:
                float_values[var] = float(raw)
            except ValueError:
                raise ValueError(
                    f"Environment variable {var} must be a float, got: {raw!r}"
                )
        else:
            float_values[var] = default
    settings = Settings(
        mcpo_base_url=source["MCPO_BASE_URL"].rstrip("/"),
        ollama_base_url=source["OLLAMA_BASE_URL"].rstrip("/"),
        ollama_model=source["OLLAMA_MODEL"],
        max_plan_retries=int_values["MAX_PLAN_RETRIES"],
        context_token_budget=int_values["CONTEXT_TOKEN_BUDGET"],
        ollama_read_timeout=int_values["OLLAMA_READ_TIMEOUT"],
        mcpo_connect_timeout=float_values["MCPO_CONNECT_TIMEOUT"],
        mcpo_read_timeout=float_values["MCPO_READ_TIMEOUT"],
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


# ── registry/operations ──
READ = "READ"
WRITE = "WRITE"
DESTRUCTIVE = "DESTRUCTIVE"


def _build_operation_entry(
    name, endpoint, risk, description, required_fields, optional_fields=None, notes=""
):
    return {
        "name": name,
        "endpoint": endpoint,
        "risk": risk,
        "description": description,
        "required_fields": required_fields,
        "optional_fields": optional_fields or {},
        "notes": notes,
    }


OPERATIONS: dict[str, dict] = {
    "obsidian_list_files_in_dir": _build_operation_entry(
        "obsidian_list_files_in_dir",
        "/obsidian/obsidian_list_files_in_dir",
        READ,
        "List files in a specific vault directory.",
        {"dirpath": "str"},
        notes="Empty directories not returned. No trailing slash on dirpath.",
    ),
    "obsidian_list_files_in_vault": _build_operation_entry(
        "obsidian_list_files_in_vault",
        "/obsidian/obsidian_list_files_in_vault",
        READ,
        "List all files in the vault root.",
        {},
    ),
    "obsidian_get_file_contents": _build_operation_entry(
        "obsidian_get_file_contents",
        "/obsidian/obsidian_get_file_contents",
        READ,
        "Get content of a single file.",
        {"filepath": "str"},
    ),
    "obsidian_simple_search": _build_operation_entry(
        "obsidian_simple_search",
        "/obsidian/obsidian_simple_search",
        READ,
        "Simple text search across all vault files.",
        {"query": "str"},
        {"context_length": ("int", 100)},
        notes="Use when user asks for notes containing specific text.",
    ),
    "obsidian_complex_search": _build_operation_entry(
        "obsidian_complex_search",
        "/obsidian/obsidian_complex_search",
        READ,
        "Complex search using JsonLogic query.",
        {"query": "dict"},
        notes="Use for structural queries (tags, folders, dates). Supports glob and regexp operators.",
    ),
    "obsidian_batch_get_file_contents": _build_operation_entry(
        "obsidian_batch_get_file_contents",
        "/obsidian/obsidian_batch_get_file_contents",
        READ,
        "Get contents of multiple files concatenated with headers.",
        {"filepaths": "list[str]"},
        notes="Prefer over multiple obsidian_get_file_contents calls.",
    ),
    "obsidian_get_periodic_note": _build_operation_entry(
        "obsidian_get_periodic_note",
        "/obsidian/obsidian_get_periodic_note",
        READ,
        "Get current periodic note for a period type.",
        {"period": "str"},
        notes="period must be: daily, weekly, monthly, quarterly, yearly.",
    ),
    "obsidian_get_recent_periodic_notes": _build_operation_entry(
        "obsidian_get_recent_periodic_notes",
        "/obsidian/obsidian_get_recent_periodic_notes",
        READ,
        "Get recent periodic notes for a period type.",
        {"period": "str"},
        {"limit": ("int", 5), "include_content": ("bool", False)},
    ),
    "obsidian_get_recent_changes": _build_operation_entry(
        "obsidian_get_recent_changes",
        "/obsidian/obsidian_get_recent_changes",
        READ,
        "Get recently modified files.",
        {},
        {"limit": ("int", 10), "days": ("int", 90)},
        notes="Requires Dataview plugin. Uses DQL query internally.",
    ),
    "obsidian_patch_content": _build_operation_entry(
        "obsidian_patch_content",
        "/obsidian/obsidian_patch_content",
        WRITE,
        "Insert content relative to a heading, block, or frontmatter field.",
        {
            "filepath": "str",
            "operation": "str",
            "target_type": "str",
            "target": "str",
            "content": "str",
        },
        notes="Prefer over append when a location anchor is specified. prepend+frontmatter is invalid.",
    ),
    "obsidian_append_content": _build_operation_entry(
        "obsidian_append_content",
        "/obsidian/obsidian_append_content",
        WRITE,
        "Append content to end of a file.",
        {"filepath": "str", "content": "str"},
        notes="Use only when no location anchor is specified.",
    ),
    "obsidian_delete_file": _build_operation_entry(
        "obsidian_delete_file",
        "/obsidian/obsidian_delete_file",
        DESTRUCTIVE,
        "Delete a file or directory from the vault.",
        {"filepath": "str", "confirm": "bool"},
        notes="confirm must be true. Planner must not include unless user says delete/remove.",
    ),
    "obsidian_get_tags": _build_operation_entry(
        "obsidian_get_tags",
        "/obsidian/obsidian_get_tags",
        READ,
        "Get all tags in the vault with counts.",
        {},
        notes="Returns inline (#tag) and frontmatter tags with hierarchical parent counts.",
    ),
}


def get_operation(name: str) -> dict | None:
    return OPERATIONS.get(name)


# ── llm/client ──
ResponseModel = TypeVar("ResponseModel", bound=BaseModel)


def _strip_model_tags(content: str) -> str:
    return re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()


def _extract_json_from_markdown(content: str) -> str:
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", content)
    return match.group(1).strip() if match else content


def _parse_with_schema(raw: str, schema: Type[ResponseModel]) -> ResponseModel | None:
    try:
        return schema.model_validate_json(raw)
    except Exception:
        return None


def create_llm(settings: Settings, temperature: float = 0.0):
    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
        temperature=temperature,
        num_predict=-1,
        timeout=settings.ollama_read_timeout,
    )


def invoke_structured(
    llm, schema: Type[ResponseModel], messages: list
) -> ResponseModel:
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
    response = llm.invoke(messages)
    content = _strip_model_tags(response.content)
    raw_json = _extract_json_from_markdown(content)
    parsed = _parse_with_schema(raw_json, schema)
    if parsed is not None:
        logger.info("Manual JSON parse succeeded for %s", schema.__name__)
        return parsed
    obj_match = re.search(r"\{[\s\S]*\}", content)
    if obj_match:
        parsed = _parse_with_schema(obj_match.group(0), schema)
        if parsed is not None:
            logger.info("Regex JSON extraction succeeded for %s", schema.__name__)
            return parsed
    raise ValueError(
        f"Failed to parse LLM output into {schema.__name__}. Raw content: {content[:500]}"
    )


# ── obsidian/api_client ──
class _TransientHTTPError(Exception):
    pass


def _is_transient(exc: BaseException) -> bool:
    return isinstance(exc, (httpx.TimeoutException, _TransientHTTPError))


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception(_is_transient),
    reraise=True,
)
def _retryable_post(
    client: httpx.Client, endpoint: str, inputs: dict
) -> httpx.Response:
    response = client.post(endpoint, json=inputs)
    if response.status_code in (429, 503):
        raise _TransientHTTPError(f"HTTP {response.status_code}")
    response.raise_for_status()
    return response


class ObsidianClient:
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
        op = get_operation(operation_name)
        if op is None:
            raise ValueError(f"Unknown operation: {operation_name}")
        endpoint = op["endpoint"]
        risk = op["risk"]
        logger.info(
            "Calling %s at %s with inputs: %s",
            operation_name,
            endpoint,
            list(inputs.keys()),
        )
        try:
            if risk == "READ":
                response = _retryable_post(self.client, endpoint, inputs)
            else:
                response = self.client.post(endpoint, json=inputs)
                if response.status_code in (429, 503):
                    logger.warning(
                        "%s returned %d -- failure may be transient but will not be retried (risk=%s)",
                        operation_name,
                        response.status_code,
                        risk,
                    )
                response.raise_for_status()
        except httpx.TimeoutException as e:
            raise httpx.TimeoutException(
                f"mcpo proxy timed out for {operation_name}: {e}"
            ) from e
        except httpx.ConnectError as e:
            raise httpx.ConnectError(
                f"mcpo proxy unreachable for {operation_name}: {e}"
            ) from e
        try:
            data = response.json()
        except Exception:
            data = response.text
        logger.info("Response from %s: status=%d", operation_name, response.status_code)
        return {"success": True, "data": data, "status_code": response.status_code}

    def health_check(self) -> bool:
        try:
            response = self.client.get("/")
            return response.status_code < 400
        except Exception as e:
            logger.error("mcpo health check failed: %s", e)
            return False

    def close(self) -> None:
        self.client.close()


# ── nodes/intent_classifier ──
INTENT_CLASSIFIER_SYSTEM_PROMPT = """You are the Intent Classification Node -- the first and only classification gate in a
multi-step Obsidian vault agent pipeline. Your output routes the entire pipeline: a
wrong classification sends all downstream nodes down the wrong branch and cannot be
corrected later.

## Your Input
You receive:
1. A system prompt (this message).
2. Up to 4 prior conversation turns (HumanMessage / AIMessage pairs) for context.
3. The current user message as the final HumanMessage.

## Your Task
Classify the current user message into EXACTLY ONE of the following intents.

### Intent Definitions

- **vault_read**: The user wants to retrieve, search, or learn from vault content.
  Use this when the user asks about a topic, concept, or subject area where they
  plausibly have notes (e.g., "what is X", "explain X", "show me X", "review X").
  Also use this for periodic note reads and recent-change queries if the user does
  not phrase them as a creation or deletion request.

- **vault_write**: The user wants to create, modify, append, or update vault content.

- **vault_delete**: The user wants to delete or remove vault content. Use ONLY when
  the user explicitly says "delete" or "remove".

- **general_qa**: The user's question does NOT require a new vault search. Use this for:
  (a) Universal factual queries (arithmetic, well-known definitions) with no plausible
      link to personal notes.
  (b) Follow-up questions that reference the PREVIOUS assistant response -- e.g., "the
      sources you found", "what you mentioned", "based on what you showed me", "how do
      I use that", "can you explain that further". These are answered from conversation
      context; a new vault search would be redundant.
  Decision rule when unsure: if conversation history is present and the user is building
  on existing results, choose general_qa. Only choose vault_read when the user clearly
  wants NEW vault content.

- **periodic_note**: The user is specifically referencing a periodic note (daily, weekly,
  monthly, quarterly, yearly) -- whether to read, create, or append.

- **recent_changes**: The user wants to know what files or content changed recently in
  the vault.

## Worked Example (hardest boundary: general_qa vs vault_read)

Prior assistant message: "Here are 3 notes I found on productivity systems: ..."
User message: "Can you explain the second one in more detail?"
Correct intent: **general_qa**
Reasoning: The user is asking for elaboration on content already returned -- no new
vault search is needed. The answer is in the conversation context.

---

User message: "Do you have any notes on the Pomodoro technique?"
Correct intent: **vault_read**
Reasoning: This is a fresh query for vault content; no prior results are being
referenced.

## Output Format
Respond with a JSON object. The "intent" field MUST be one of the six strings listed
above. The "reasoning" field should be a single concise sentence (technical register,
<=40 words) stating why this intent was selected over the closest alternative.

```json
{
  "intent": "<one of: vault_read | vault_write | vault_delete | general_qa | periodic_note | recent_changes>",
  "reasoning": "<one sentence, <=40 words>"
}
```

Think step by step before producing the JSON. Do not emit any text outside the JSON block."""


def intent_classifier(state: AgentState, llm) -> dict:
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

    user_message = state.get("user_message", "")
    conversation_history = state.get("conversation_history", [])
    messages = [SystemMessage(content=INTENT_CLASSIFIER_SYSTEM_PROMPT)]
    for turn in conversation_history[-CONVERSATION_WINDOW_TURNS:]:
        role = turn.get("role", "user")
        content = turn.get("content", "")
        if role == "user":
            messages.append(HumanMessage(content=content))
        else:
            messages.append(AIMessage(content=content))
    messages.append(HumanMessage(content=user_message))
    try:
        result = invoke_structured(llm, IntentOutput, messages)
        logger.info(
            "Intent classified: %s (reasoning: %s)",
            result.intent,
            result.reasoning[:100],
        )
        return {"intent": result.model_dump()}
    except Exception as e:
        logger.error("Intent classification failed: %s", e)
        fallback = IntentOutput(
            intent="vault_read", reasoning=f"Fallback due to classification error: {e}"
        )
        return {
            "intent": fallback.model_dump(),
            "error": f"[intent_classifier] Classification failed, defaulting to vault_read: {e}",
        }


# ── nodes/web_search_extractor ──
def web_search_extractor(state: AgentState) -> dict:
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
            "error": make_node_error("web_search_extractor", e),
        }


# ── nodes/kb_retrieval_agent ──
def _detect_period(message: str) -> str:
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
    noise = {
        "what",
        "where",
        "when",
        "how",
        "who",
        "which",
        "can",
        "could",
        "would",
        "should",
        "do",
        "does",
        "did",
        "is",
        "are",
        "was",
        "were",
        "the",
        "a",
        "an",
        "my",
        "me",
        "i",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "with",
        "about",
        "find",
        "search",
        "look",
        "get",
        "show",
        "list",
        "tell",
        "give",
        "have",
        "has",
        "notes",
        "note",
        "vault",
        "obsidian",
        "please",
        "thanks",
        "task",
        "analyze",
        "chat",
        "history",
        "context",
    }
    words = message.lower().split()
    terms = [w for w in words if w not in noise and len(w) > 2]
    return " ".join(terms[:5])


def kb_retrieval_agent(state: AgentState, obsidian_client: ObsidianClient) -> dict:
    intent_data = state.get("intent")
    intent = get_intent(intent_data)
    user_message = state.get("user_message", "")
    if intent == "general_qa":
        return {"kb_results": []}
    if user_message.startswith("###"):
        return {"kb_results": []}
    try:
        if intent == "recent_changes":
            result = obsidian_client.call_tool("obsidian_get_recent_changes", {})
            return {
                "kb_results": [{"source": "recent_changes", "data": result.get("data")}]
            }
        if intent == "periodic_note":
            period = _detect_period(user_message)
            result = obsidian_client.call_tool(
                "obsidian_get_periodic_note", {"period": period}
            )
            return {
                "kb_results": [
                    {
                        "source": "periodic_note",
                        "period": period,
                        "data": result.get("data"),
                    }
                ]
            }
        logger.debug("KB retrieval: intent '%s' uses default search path", intent)
        search_query = _extract_search_terms(user_message)
        if search_query:
            result = obsidian_client.call_tool(
                "obsidian_simple_search", {"query": search_query, "context_length": 150}
            )
            return {
                "kb_results": [
                    {
                        "source": "simple_search",
                        "query": search_query,
                        "data": result.get("data"),
                    }
                ]
            }
        return {"kb_results": []}
    except Exception as e:
        logger.error("KB retrieval failed: %s", e)
        return {"kb_results": [], "error": make_node_error("kb_retrieval_agent", e)}


# ── nodes/file_extraction_agent ──
def _maybe_decode_data_uri(data, name: str) -> str:
    if not isinstance(data, str) or not data.startswith("data:"):
        return str(data)
    _, _, encoded = data.partition(",")
    try:
        return base64.b64decode(encoded).decode("utf-8", errors="replace")
    except Exception as decode_err:
        logger.warning("base64 decode failed for attachment '%s': %s", name, decode_err)
        return "[binary content -- could not decode]"


def file_extraction_agent(state: AgentState) -> dict:
    attachments = state.get("file_attachments", [])
    if not attachments:
        return {"file_contents": []}
    try:
        contents = []
        for attachment in attachments:
            name = attachment.get("name", attachment.get("filename", "unknown"))
            data = attachment.get("data", attachment.get("content", ""))
            contents.append(
                {"filename": name, "content": _maybe_decode_data_uri(data, name)[:5000]}
            )
        logger.info("Extracted content from %d file attachments", len(contents))
        return {"file_contents": contents}
    except Exception as e:
        logger.error("File extraction failed: %s", e)
        return {
            "file_contents": [],
            "error": make_node_error("file_extraction_agent", e),
        }


# ── nodes/image_description_agent ──
def _describe_single_image(url: str, llm) -> str:
    from langchain_core.messages import HumanMessage

    message = HumanMessage(
        content=[
            {"type": "text", "text": "Describe this image concisely in 1-2 sentences."},
            {"type": "image_url", "image_url": {"url": url}},
        ]
    )
    try:
        return llm.invoke([message]).content
    except Exception as e:
        logger.warning("Failed to describe image: %s", e)
        return "[Image description unavailable]"


def _collect_image_descriptions(images: list, llm) -> list[str]:
    descriptions = []
    for image in images:
        url = image.get("url", image.get("data", ""))
        if url:
            descriptions.append(_describe_single_image(url, llm))
    return descriptions


def image_description_agent(state: AgentState, llm) -> dict:
    images = state.get("image_attachments", [])
    if not images:
        return {"image_descriptions": []}
    try:
        descriptions = _collect_image_descriptions(images, llm)
        logger.info("Described %d images", len(descriptions))
        return {"image_descriptions": descriptions}
    except Exception as e:
        logger.error("Image description failed: %s", e)
        return {
            "image_descriptions": [],
            "error": make_node_error("image_description_agent", e),
        }


# ── nodes/context_aggregator ──
def _apply_budget_section(
    source, formatter, threshold: int, header: str, sections: list, remaining: int
) -> int:
    if not source or remaining <= threshold:
        return remaining
    text = formatter(source)
    chars = min(len(text), remaining)
    sections.append(f"### {header}\n{text[:chars]}")
    return remaining - chars


def _format_history(turns: list[dict]) -> str:
    lines = []
    for turn in turns:
        role = turn.get("role", "unknown")
        content = turn.get("content", "")
        if isinstance(content, list):
            content = " ".join(
                item.get("text", "")
                for item in content
                if isinstance(item, dict) and item.get("type") == "text"
            )
        lines.append(f"**{role}**: {content}")
    return "\n".join(lines)


def _format_kb_item(item) -> list[str]:
    if not isinstance(item, dict):
        return [f"- {str(item)[:200]}"]
    filename = item.get("filename", item.get("path", ""))
    lines = [f"- `{filename}`"]
    for match in item.get("matches", [])[:3]:
        ctx = match.get("context", "")
        if ctx:
            lines.append(f"  > {ctx[:200]}")
    return lines


def _format_kb_list(data: list) -> list[str]:
    lines = []
    for item in data[:10]:
        lines.extend(_format_kb_item(item))
    return lines


def _format_kb_results(results: list[dict]) -> str:
    lines = []
    for kb_result in results:
        data = kb_result.get("data", "")
        if isinstance(data, list):
            lines.extend(_format_kb_list(data))
        elif isinstance(data, str):
            lines.append(data[:2000])
        else:
            lines.append(json.dumps(data, indent=2)[:2000])
    return "\n".join(lines)


def _format_file_contents(contents: list[dict]) -> str:
    lines = []
    for file_entry in contents:
        name = file_entry.get("filename", "unknown")
        content = file_entry.get("content", "")
        lines.append(f"**{name}**:\n{content[:1000]}")
    return "\n\n".join(lines)


def context_aggregator(
    state: AgentState, token_budget: int, chars_per_token: float
) -> dict:
    char_budget = int(token_budget * chars_per_token)
    sections: list[str] = []
    remaining = char_budget
    history = state.get("conversation_history", [])
    remaining = _apply_budget_section(
        history[-CONVERSATION_WINDOW_TURNS:] if history else [],
        _format_history,
        0,
        "Conversation Context",
        sections,
        remaining,
    )
    remaining = _apply_budget_section(
        state.get("kb_results", []),
        _format_kb_results,
        200,
        "Vault Search Results",
        sections,
        remaining,
    )
    remaining = _apply_budget_section(
        state.get("file_contents", []),
        _format_file_contents,
        200,
        "File Attachments",
        sections,
        remaining,
    )
    remaining = _apply_budget_section(
        state.get("web_search_context", []),
        lambda x: "\n".join(x),
        100,
        "Web Search Results",
        sections,
        remaining,
    )
    remaining = _apply_budget_section(
        state.get("image_descriptions", []),
        lambda x: "\n".join(x),
        50,
        "Image Descriptions",
        sections,
        remaining,
    )
    aggregated = "\n\n".join(sections)
    logger.info(
        "Aggregated context: %d chars (%d estimated tokens, budget %d)",
        len(aggregated),
        int(len(aggregated) / chars_per_token),
        token_budget,
    )
    return {"aggregated_context": aggregated}


# ── nodes/planner ──
PLANNER_SYSTEM_PROMPT_TEMPLATE = """You are the Planner Node -- Step 6 of 10 in an Obsidian vault agent pipeline. Your
output is the ONLY plan the executor will run. There is no human review between your
plan and live vault operations. Plan conservatively: an unnecessary WRITE or DESTRUCTIVE
call has real consequences on the user's vault and cannot be automatically undone.

Your plan may be validated and returned to you with errors (up to 2 retries). When
retry errors are present in the input, you MUST fix every listed error -- do not
reproduce any rejected call unchanged.

## Your Inputs (injected in the HumanMessage that follows this system prompt)
- **User Request**: The original user message.
- **Detected Intent**: One of vault_read | vault_write | vault_delete | general_qa |
  periodic_note | recent_changes, classified by the prior node.
- **Context** (when present): Aggregated vault search results, file contents, image
  descriptions, and/or web search results -- all token-budget-capped.
- **Validation Errors** (on retry only): A list of constraint violations from the
  previous plan attempt.

## Available Operations
{operations_summary}

## Planning Rules
1. Maximum 10 operations per plan.
2. NEVER plan DESTRUCTIVE operations unless the user's message explicitly contains
   "delete" or "remove".
3. Prefer writing only to file paths discovered via a READ in this plan or explicitly
   named by the user. This guideline is not enforced by the validator -- it is a safety
   guardrail against accidental overwrites.
4. Every operation MUST have a non-empty rationale (one declarative sentence explaining
   operational necessity).
5. Prefer obsidian_batch_get_file_contents over multiple obsidian_get_file_contents calls.
6. Prefer obsidian_patch_content over obsidian_append_content when the user specifies
   a target location (heading, block, or frontmatter field).
7. Paths are vault-relative, forward slashes, no leading slash. Paths are CASE-SENSITIVE.
8. For general_qa intent, return an empty calls list immediately.

## Output Format
Think step by step in reasoning_trace first, then produce the JSON. Keep reasoning_trace
to 3-5 sentences: name the intent, list the operations needed and why, and state any
safety decisions made. Keep each rationale to one sentence.

```json
{{
  "calls": [
    {{
      "operation_name": "<must match a key in Available Operations>",
      "inputs": {{ "<field>": "<value>" }},
      "rationale": "<one declarative sentence>",
      "risk": "<READ | WRITE | DESTRUCTIVE>"
    }}
  ],
  "reasoning_trace": "<3-5 sentences of step-by-step reasoning>"
}}
```

Do not emit any text outside the JSON block."""


def _build_operations_summary() -> str:
    lines = []
    for name, op in OPERATIONS.items():
        req = ", ".join(f"{k}: {v}" for k, v in op["required_fields"].items())
        opt = ", ".join(
            f"{k} (default={v[1]})" for k, v in op.get("optional_fields", {}).items()
        )
        fields = req
        if opt:
            fields += f" | optional: {opt}"
        lines.append(
            f"- **{name}** [{op['risk']}]: {op['description']} Fields: {fields or 'none'}"
        )
    return "\n".join(lines)


def planner(state: AgentState, llm) -> dict:
    from langchain_core.messages import HumanMessage, SystemMessage

    user_message = state.get("user_message", "")
    intent_data = state.get("intent")
    intent = get_intent(intent_data)
    aggregated_context = state.get("aggregated_context", "")
    validation_errors = state.get("plan_validation_errors", [])
    retry_count = state.get("retry_count", 0)
    existing_error = state.get("error", "")
    if existing_error and retry_count == 0:
        logger.warning("Planner invoked with existing error state: %s", existing_error)
    system_prompt = PLANNER_SYSTEM_PROMPT_TEMPLATE.format(
        operations_summary=_build_operations_summary()
    )
    messages = [SystemMessage(content=system_prompt)]
    user_prompt_parts = [f"## User Request\n{user_message}"]
    user_prompt_parts.append(f"\n## Detected Intent: {intent}")
    if aggregated_context:
        user_prompt_parts.append(f"\n## Context\n{aggregated_context}")
    if validation_errors and retry_count > 0:
        errors_text = "\n".join(f"- {e}" for e in validation_errors)
        user_prompt_parts.append(
            f"\n## VALIDATION ERRORS FROM PREVIOUS ATTEMPT (fix ALL of these)\n{errors_text}"
        )
        prev_plan = state.get("plan")
        if prev_plan:
            user_prompt_parts.append(
                f"\n## Previous rejected plan (for reference)\n```json\n{json.dumps(prev_plan,indent=2)}\n```"
            )
    messages.append(HumanMessage(content="\n".join(user_prompt_parts)))
    try:
        result = invoke_structured(llm, PlanOutput, messages)
        logger.info(
            "Planner produced %d calls (retry=%d)", len(result.calls), retry_count
        )
        return {"plan": result.model_dump(), "retry_count": retry_count + 1}
    except Exception as e:
        logger.error("Planner failed: %s", e)
        empty = PlanOutput(calls=[], reasoning_trace=f"Planner error: {e}")
        return {
            "plan": empty.model_dump(),
            "retry_count": retry_count + 1,
            "error": make_node_error("planner", e),
        }


# ── nodes/plan_validator ──
def _is_safe_vault_path(path: str) -> bool:
    if ".." in path:
        return False
    if path.startswith("/") or path.startswith("\\"):
        return False
    if ":" in path:
        return False
    return True


def _validate_destructive_call(
    i: int, op_name: str, inputs: dict, user_lower: str
) -> list[str]:
    errors = []
    if "delete" not in user_lower and "remove" not in user_lower:
        errors.append(
            f"Call {i} ({op_name}): destructive operation requires 'delete' or 'remove' in user message"
        )
    filepath = inputs.get("filepath", "")
    if filepath and not filepath.endswith(".md"):
        if "folder" not in user_lower and "directory" not in user_lower:
            errors.append(
                f"Call {i} ({op_name}): filepath '{filepath}' must end in .md for delete operations unless folder deletion is explicitly intended"
            )
    return errors


def _validate_paths(i: int, op_name: str, inputs: dict) -> list[str]:
    errors = []
    for field_name in ("filepath", "dirpath"):
        path_val = inputs.get(field_name, "")
        if path_val and not _is_safe_vault_path(path_val):
            errors.append(
                f"Call {i} ({op_name}): invalid path: traversal detected in '{field_name}' value '{path_val}'"
            )
    return errors


def _validate_call(i: int, call: dict, user_lower: str) -> list[str]:
    errors = []
    op_name = call.get("operation_name", "")
    inputs = call.get("inputs", {})
    rationale = call.get("rationale", "")
    op_entry = OPERATIONS.get(op_name)
    if op_entry is None:
        return [f"Call {i}: unknown operation '{op_name}'"]
    for field_name in op_entry["required_fields"]:
        if field_name not in inputs:
            errors.append(f"Call {i} ({op_name}): missing required field: {field_name}")
    if op_entry["risk"] == "DESTRUCTIVE":
        errors.extend(_validate_destructive_call(i, op_name, inputs, user_lower))
    if not rationale or not rationale.strip():
        errors.append(f"Call {i} ({op_name}): empty rationale")
    if op_name == "obsidian_delete_file" and inputs.get("confirm") is not True:
        errors.append(f"Call {i} ({op_name}): confirm must be true")
    errors.extend(_validate_paths(i, op_name, inputs))
    return errors


def plan_validator(state: AgentState) -> dict:
    plan = state.get("plan") or {}
    calls = plan.get("calls", [])
    user_message = state.get("user_message", "")
    user_lower = user_message.lower()
    errors: list[str] = []
    if not calls:
        logger.info("Plan validator: empty plan, valid")
        return {"plan_valid": True, "plan_validation_errors": []}
    if len(calls) > 10:
        errors.append(f"Plan exceeds maximum of 10 calls (has {len(calls)})")
    for i, call in enumerate(calls):
        errors.extend(_validate_call(i, call, user_lower))
    plan_valid = len(errors) == 0
    if plan_valid:
        logger.info("Plan validated: %d calls, all rules passed", len(calls))
    else:
        logger.warning("Plan validation failed with %d errors: %s", len(errors), errors)
    return {"plan_valid": plan_valid, "plan_validation_errors": errors}


# ── nodes/executor ──
def _record_and_maybe_halt(
    results: list, op_name: str, inputs: dict, error_detail: str, risk: str
) -> dict | None:
    results.append(
        ExecutionResult(
            operation_name=op_name,
            inputs=inputs,
            response=None,
            success=False,
            error=error_detail,
        )
    )
    if risk in ("WRITE", "DESTRUCTIVE"):
        return {
            "execution_results": results,
            "error": f"[executor] {op_name} failed ({risk}): {error_detail}",
        }
    return None


def _execute_single_call(
    call: dict,
    call_index: int,
    total: int,
    results: list,
    obsidian_client: ObsidianClient,
) -> dict | None:
    op_name = call.get("operation_name", "")
    inputs = call.get("inputs", {})
    risk = call.get("risk", "READ")
    logger.info("Executing call %d/%d: %s (%s)", call_index + 1, total, op_name, risk)
    try:
        response = obsidian_client.call_tool(op_name, inputs)
        results.append(
            ExecutionResult(
                operation_name=op_name,
                inputs=inputs,
                response=response.get("data"),
                success=True,
            )
        )
        logger.info("Call %d succeeded: %s", call_index + 1, op_name)
        return None
    except httpx.HTTPStatusError as e:
        error_detail = f"HTTP {e.response.status_code}: {e.response.text[:200]}"
        logger.error("Call %d failed: %s - %s", call_index + 1, op_name, error_detail)
        return _record_and_maybe_halt(results, op_name, inputs, error_detail, risk)
    except Exception as e:
        error_detail = f"{type(e).__name__}: {e}"
        logger.error("Call %d failed: %s - %s", call_index + 1, op_name, error_detail)
        return _record_and_maybe_halt(results, op_name, inputs, error_detail, risk)


def executor(state: AgentState, obsidian_client: ObsidianClient) -> dict:
    plan = state.get("plan") or {}
    calls = plan.get("calls") or []
    if not calls:
        return {"execution_results": []}
    results = []
    for call_index, call in enumerate(calls):
        halt = _execute_single_call(
            call, call_index, len(calls), results, obsidian_client
        )
        if halt:
            return halt
    return {"execution_results": results}


# ── nodes/summarizer ──
SUMMARIZER_SYSTEM_PROMPT = """You are the Summarizer Node -- the final step in an Obsidian vault agent pipeline.
Your output is the exact text the user reads. You receive structured execution results
and aggregated context from prior nodes; your job is to translate these into a polished,
user-facing natural language response.

IMPORTANT: Your response will be ASCII-encoded before delivery. Use only ASCII
characters. Do not use Unicode quotes, em-dashes, ellipses, or non-ASCII symbols.

## Your Inputs (injected in the HumanMessage that follows this system prompt)
- **User Request**: The original user message.
- **Context** (when present): Aggregated vault content, conversation history, and/or
  web search results -- pre-formatted and token-capped.
- **Execution Results** (when present): A Markdown bullet list of vault operations
  performed, each with status (OK or FAILED) and a response or error excerpt.

## Response Mode

### Mode A -- Vault Operations Were Executed
Use when Execution Results are present.

Rules:
- Acknowledge every file path touched and every operation performed, with its outcome.
- Summarize relevant vault content found; do NOT reproduce large blocks of raw note text.
  Quote at most 2-3 lines from any single note, inside a code fence.
- Use code fences for all file paths and note content excerpts.
- Do not use headers above H3 (###).
- Do not mention internal state fields, retry counts, validation errors, or pipeline
  mechanics.
- Register: plain, direct language for a knowledge-worker audience. No unnecessary
  technical jargon unless the user's request is itself technical.

Mode A template:
```
### Summary
[1-2 sentences describing what was done and the overall outcome.]

### Details
- `path/to/file.md` -- [operation] [OK/FAILED]: [one-line outcome or error]
- `path/to/other.md` -- [operation] [OK]: [brief content excerpt if relevant]
  ```
  [excerpt, max 3 lines]
  ```
```

### Mode B -- No Vault Operations (general_qa or conversational follow-up)
Use when no Execution Results are present.

Rules:
- Answer the user's question using the provided Context and conversation history.
- Do not invent vault content or imply a vault search was performed.
- Do not use headers; plain prose or a brief bullet list is appropriate.
- Do not output raw JSON, JSON arrays, or code blocks with structured data -- use plain
  prose or Markdown bullet lists only.
- If you want to ask clarifying questions, write them as a numbered Markdown list
  (e.g., "1. What aspect of X interests you most?"), never as a JSON array.
- Register: conversational but precise. Match the user's register (technical query ->
  technical answer; casual query -> casual answer).
- If no relevant context is available, say so directly and offer to search the vault.

Mode B template:
```
[Direct answer to the user's question, 2-5 sentences or a concise bullet list.]
[If context was used: "Based on [source], ..." to attribute the answer.]
```

## Output Rules (both modes)
- ASCII-only characters throughout.
- No text outside the response body (no preamble like "Here is my response:").
- No mention of internal pipeline state, node names, retry history, or validation errors.
- Do not reference or reproduce source ID markup (e.g., "source id=2", "<source id=N>")
  in your response. These are internal context-formatting artifacts invisible to the user."""


def _format_list_item_summ(item) -> list[str]:
    if not isinstance(item, dict):
        return [f"    - {str(item)[:200]}"]
    fname = item.get("filename", item.get("path", "?"))
    lines = [f"    - `{fname}`"]
    for m in item.get("matches", [])[:2]:
        ctx = m.get("context", "")
        if ctx:
            lines.append(f"      > {ctx[:200]}")
    return lines


def _format_list_response_summ(resp: list) -> list[str]:
    if not resp:
        return ["  No results found"]
    lines = [f"  Returned {len(resp)} results:"]
    for item in resp[:5]:
        lines.extend(_format_list_item_summ(item))
    return lines


def _format_success_response(resp) -> list[str]:
    if isinstance(resp, str):
        return [f"  Response: {resp[:500]}"]
    if isinstance(resp, list):
        return _format_list_response_summ(resp)
    if isinstance(resp, dict):
        return [f"  Response: {json.dumps(resp,indent=2)[:500]}"]
    return []


def _format_execution_results(results: list[dict]) -> str:
    lines = []
    for result in results:
        op = result.get("operation_name", "unknown")
        success = result.get("success", False)
        status = "OK" if success else "FAILED"
        inputs = result.get("inputs", {})
        if not isinstance(inputs, dict):
            inputs = {}
        filepath = inputs.get(
            "filepath", inputs.get("dirpath", inputs.get("query", ""))
        )
        lines.append(f"- **{op}** [{status}] target: `{filepath}`")
        if success and result.get("response"):
            lines.extend(_format_success_response(result["response"]))
        if not success and result.get("error"):
            lines.append(f"  Error: {result['error']}")
    return "\n".join(lines)


def _build_fallback_response(user_message: str, results: list[dict]) -> str:
    lines = [
        "I processed your request but encountered an issue generating a summary.\n"
    ]
    if results:
        lines.append("### Operations performed:")
        for r in results:
            op = r.get("operation_name", "unknown")
            success = "succeeded" if r.get("success") else "failed"
            lines.append(f"- `{op}`: {success}")
    return "\n".join(lines)


def summarizer(state: AgentState, llm) -> dict:
    from langchain_core.messages import HumanMessage, SystemMessage

    user_message = state.get("user_message", "")
    execution_results = state.get("execution_results", [])
    aggregated_context = state.get("aggregated_context", "")
    messages = [SystemMessage(content=SUMMARIZER_SYSTEM_PROMPT)]
    parts = [f"## User Request\n{user_message}"]
    if aggregated_context:
        parts.append(f"\n## Context\n{aggregated_context}")
    if execution_results:
        results_text = _format_execution_results(execution_results)
        parts.append(f"\n## Execution Results\n{results_text}")
    messages.append(HumanMessage(content="\n".join(parts)))
    try:
        response = llm.invoke(messages)
        content = _strip_model_tags(response.content)
        logger.info("Summarizer produced %d char response", len(content))
        return {"final_response": content}
    except Exception as e:
        logger.error("Summarizer failed (%s): %s", type(e).__name__, e)
        fallback = _build_fallback_response(user_message, execution_results)
        return {"final_response": fallback}


# ── nodes/error_handler ──
FALLBACK_MESSAGE = "An unexpected error occurred while processing your request. Please try rephrasing or simplifying your request."


def _sanitize_error(error: str) -> str:
    sanitized = re.sub(r"\[[\w_]+\]\s*", "", error)
    if len(sanitized) > 300:
        sanitized = sanitized[:300] + "..."
    return sanitized


def error_handler(state: AgentState) -> dict:
    error = state.get("error", "")
    user_message = state.get("user_message", "")
    retry_count = state.get("retry_count", 0)
    plan_errors = state.get("plan_validation_errors", [])
    try:
        parts = []
        if user_message:
            parts.append(
                f'I was unable to complete your request: "{user_message[:200]}"'
            )
        if plan_errors:
            parts.append(
                f"\nThe operation plan could not be validated after {retry_count} attempt(s). The planner was unable to produce a valid set of operations for your request."
            )
        elif error:
            sanitized = _sanitize_error(error)
            parts.append(f"\nThe operation failed: {sanitized}")
        else:
            parts.append("\nAn unexpected error occurred during processing.")
        parts.append(
            "\nYou may try:\n- Rephrasing your request more specifically\n- Breaking a complex request into simpler steps\n- Checking that the files or folders you referenced exist in your vault"
        )
        response = "\n".join(parts)
        logger.info("Error handler produced response (%d chars)", len(response))
        return {"final_response": response}
    except Exception as e:
        logger.error("Error handler itself failed: %s", e)
        return {"final_response": FALLBACK_MESSAGE}


# ── graph/builder ──
def _make_safe(node_fn, node_name: str):
    def safe_node(state: AgentState) -> dict:
        try:
            return node_fn(state)
        except Exception as e:
            logger.error("[%s] Unhandled error: %s", node_name, e, exc_info=True)
            return {"error": f"[{node_name}] {type(e).__name__}: {e}"}

    safe_node.__name__ = node_name
    return safe_node


def _route_after_validation(state: AgentState, max_retries: int) -> str:
    if state.get("plan_valid", False):
        return "executor"
    if state.get("retry_count", 0) < max_retries:
        return "planner"
    return "error_handler"


def _route_after_execution(state: AgentState) -> str:
    return "error_handler" if state.get("error", "") else "summarizer"


def _create_node_closures(
    llm, obsidian_client: ObsidianClient, settings: Settings
) -> dict:
    return {
        "intent_classifier": lambda state: intent_classifier(state, llm),
        "web_search_extractor": lambda state: web_search_extractor(state),
        "kb_retrieval_agent": lambda state: kb_retrieval_agent(state, obsidian_client),
        "file_extraction_agent": lambda state: file_extraction_agent(state),
        "image_description_agent": lambda state: image_description_agent(state, llm),
        "context_aggregator": lambda state: context_aggregator(
            state, settings.context_token_budget, settings.chars_per_token
        ),
        "planner": lambda state: planner(state, llm),
        "plan_validator": lambda state: plan_validator(state),
        "executor": lambda state: executor(state, obsidian_client),
        "summarizer": lambda state: summarizer(state, llm),
        "error_handler": lambda state: error_handler(state),
    }


def build_graph(settings: Settings, obsidian_client: ObsidianClient):
    from langgraph.graph import StateGraph, START, END

    llm = create_llm(settings)
    nodes = _create_node_closures(llm, obsidian_client, settings)
    builder = StateGraph(AgentState)
    for name, fn in nodes.items():
        builder.add_node(name, _make_safe(fn, name))
    builder.add_edge(START, "intent_classifier")
    builder.add_edge("intent_classifier", "web_search_extractor")
    builder.add_edge("intent_classifier", "kb_retrieval_agent")
    builder.add_edge("intent_classifier", "file_extraction_agent")
    builder.add_edge("intent_classifier", "image_description_agent")
    builder.add_edge("web_search_extractor", "context_aggregator")
    builder.add_edge("kb_retrieval_agent", "context_aggregator")
    builder.add_edge("file_extraction_agent", "context_aggregator")
    builder.add_edge("image_description_agent", "context_aggregator")
    builder.add_edge("context_aggregator", "planner")
    builder.add_edge("planner", "plan_validator")
    builder.add_conditional_edges(
        "plan_validator",
        partial(_route_after_validation, max_retries=settings.max_plan_retries),
        {
            "executor": "executor",
            "planner": "planner",
            "error_handler": "error_handler",
        },
    )
    builder.add_conditional_edges(
        "executor",
        _route_after_execution,
        {"summarizer": "summarizer", "error_handler": "error_handler"},
    )
    builder.add_edge("summarizer", END)
    builder.add_edge("error_handler", END)
    compiled = builder.compile()
    logger.info("LangGraph pipeline compiled successfully")
    try:
        from IPython.display import Image, display

        display(Image(compiled.get_graph().draw_mermaid()))
    except Exception:
        pass
    return compiled


# ── helpers ──
def _extract_images_from_files(files: list) -> list[dict]:
    images = []
    for f in files:
        if not isinstance(f, dict):
            continue
        if f.get("type", "").startswith("image/") and f.get("url"):
            images.append({"url": f["url"]})
            continue
        file_obj = f.get("file")
        if isinstance(file_obj, dict):
            content_type = file_obj.get("meta", {}).get("content_type", "")
            url = file_obj.get("url", "")
            if content_type.startswith("image/") and url:
                images.append({"url": url})
    return images


def _classify_content_parts(content: list) -> tuple[list[str], list[dict], list]:
    text_parts: list[str] = []
    image_attachments: list[dict] = []
    file_attachments: list = []
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
    return text_parts, image_attachments, file_attachments


# ── Pipeline ──
class Pipeline:
    class Valves(BaseModel):
        pipelines: List[str] = ["*"]
        priority: int = 0
        MCPO_BASE_URL: str = Field(
            default="", description="mcpo proxy base URL (e.g. http://mcpo-1:8000)"
        )
        OLLAMA_BASE_URL: str = Field(
            default="",
            description="Ollama host URL (e.g. http://host.docker.internal:11434)",
        )
        OLLAMA_MODEL: str = Field(
            default="", description="Ollama model ID (e.g. deepseek-r1:latest)"
        )
        MAX_PLAN_RETRIES: int = Field(
            default=3, description="Planner retry ceiling before error"
        )
        CONTEXT_TOKEN_BUDGET: int = Field(
            default=4000,
            description="Token budget for context aggregation; lower values reduce LLM memory pressure during structured output generation (GGML_ASSERT errors indicate this is too high)",
        )
        OLLAMA_READ_TIMEOUT: int = Field(
            default=120, description="HTTP read timeout in seconds for LLM calls"
        )
        MCPO_CONNECT_TIMEOUT: float = Field(
            default=5.0,
            description="TCP connect timeout in seconds for mcpo proxy calls",
        )
        MCPO_READ_TIMEOUT: float = Field(
            default=30.0,
            description="HTTP read timeout in seconds for mcpo proxy calls",
        )

    def __init__(self):
        self.name = "Obsidian Agent Pipeline"
        try:
            self.valves = self.Valves(
                **{
                    "MCPO_BASE_URL": os.getenv("MCPO_BASE_URL", ""),
                    "OLLAMA_BASE_URL": os.getenv("OLLAMA_BASE_URL", ""),
                    "OLLAMA_MODEL": os.getenv("OLLAMA_MODEL", ""),
                    "MAX_PLAN_RETRIES": int(os.getenv("MAX_PLAN_RETRIES", "3")),
                    "CONTEXT_TOKEN_BUDGET": int(
                        os.getenv("CONTEXT_TOKEN_BUDGET", "4000")
                    ),
                    "OLLAMA_READ_TIMEOUT": int(os.getenv("OLLAMA_READ_TIMEOUT", "120")),
                    "MCPO_CONNECT_TIMEOUT": float(
                        os.getenv("MCPO_CONNECT_TIMEOUT", "5.0")
                    ),
                    "MCPO_READ_TIMEOUT": float(os.getenv("MCPO_READ_TIMEOUT", "30.0")),
                }
            )
        except ValueError as e:
            raise ValueError(
                f"Invalid integer environment variable for Obsidian Agent Pipeline: {e}. MAX_PLAN_RETRIES, CONTEXT_TOKEN_BUDGET, and OLLAMA_READ_TIMEOUT must be integers."
            ) from e
        self.graph = None
        self.obsidian_client = None

    async def on_startup(self):
        logger.info("Obsidian Agent Pipeline starting up...")
        try:
            env = {
                "MCPO_BASE_URL": self.valves.MCPO_BASE_URL,
                "OLLAMA_BASE_URL": self.valves.OLLAMA_BASE_URL,
                "OLLAMA_MODEL": self.valves.OLLAMA_MODEL,
                "MAX_PLAN_RETRIES": str(self.valves.MAX_PLAN_RETRIES),
                "CONTEXT_TOKEN_BUDGET": str(self.valves.CONTEXT_TOKEN_BUDGET),
                "OLLAMA_READ_TIMEOUT": str(self.valves.OLLAMA_READ_TIMEOUT),
                "MCPO_CONNECT_TIMEOUT": str(self.valves.MCPO_CONNECT_TIMEOUT),
                "MCPO_READ_TIMEOUT": str(self.valves.MCPO_READ_TIMEOUT),
            }
            settings = get_settings(env)
            self.obsidian_client = ObsidianClient(settings)
            self.graph = build_graph(settings, self.obsidian_client)
            if self.obsidian_client.health_check():
                logger.info("mcpo proxy connectivity: OK")
            else:
                logger.warning(
                    "mcpo proxy connectivity: FAILED (pipeline may not work)"
                )
            logger.info("Obsidian Agent Pipeline started successfully")
        except Exception as e:
            logger.error("Obsidian Agent Pipeline startup failed: %s", e, exc_info=True)
            raise

    async def on_shutdown(self):
        if self.obsidian_client:
            self.obsidian_client.close()
        logger.info("Obsidian Agent Pipeline shut down")

    async def inlet(self, body: dict, user: Optional[dict] = None) -> dict:
        messages = body.get("messages", [])
        if not messages:
            return body
        logger.info(
            "inlet: task=%r keys=%s messages=%d files=%d",
            body.get("task"),
            sorted(k for k in body.keys() if not k.startswith("_")),
            len(messages),
            len(body.get("files", [])),
        )
        for i, f in enumerate(body.get("files", [])[:3]):
            logger.info("inlet: file[%d]: %s", i, str(f)[:300])
        last_message = messages[-1]
        content = last_message.get("content", "")
        file_attachments = []
        image_attachments = []
        if isinstance(content, list):
            text_parts, image_attachments, file_attachments = _classify_content_parts(
                content
            )
            body["_extracted_text"] = " ".join(text_parts)
        file_upload_images = _extract_images_from_files(body.get("files", []))
        if file_upload_images:
            image_attachments = image_attachments + file_upload_images
            logger.info(
                "inlet: %d image(s) extracted from body['files']",
                len(file_upload_images),
            )
        if image_attachments:
            logger.info("inlet: total image_attachments=%d", len(image_attachments))
        body["file_attachments"] = file_attachments
        body["image_attachments"] = image_attachments
        body["raw_search_results"] = body.get("web_search_results", [])
        return body

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        try:
            if self.graph is None:
                return "Pipeline not initialized. Check startup logs for errors."
            task = body.get("task", "")
            if task:
                logger.debug("Skipping graph for OpenWebUI system task: %s", task)
                return ""
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
            return "An error occurred while processing your request. Please try rephrasing or simplifying."

    async def outlet(self, body: dict, user: Optional[dict] = None) -> dict:
        messages = body.get("messages", [])
        if messages:
            last = messages[-1]
            content = last.get("content", "")
            if isinstance(content, str):
                last["content"] = content.encode("ascii", errors="replace").decode(
                    "ascii"
                )
        return body
