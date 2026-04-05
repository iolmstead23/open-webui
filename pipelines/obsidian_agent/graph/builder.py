"""Graph builder -- assembles and compiles the LangGraph StateGraph.

Dependencies: config, state, nodes.
Exposed interface: build_graph() factory function.
"""
import logging

from langgraph.graph import StateGraph, START, END

from obsidian_agent.config import Settings
from obsidian_agent.state.schema import AgentState
from obsidian_agent.llm.client import create_llm
from obsidian_agent.obsidian.api_client import ObsidianClient

from obsidian_agent.nodes.intent_classifier import intent_classifier
from obsidian_agent.nodes.web_search_extractor import web_search_extractor
from obsidian_agent.nodes.kb_retrieval_agent import kb_retrieval_agent
from obsidian_agent.nodes.file_extraction_agent import file_extraction_agent
from obsidian_agent.nodes.image_description_agent import image_description_agent
from obsidian_agent.nodes.context_aggregator import context_aggregator
from obsidian_agent.nodes.planner import planner
from obsidian_agent.nodes.plan_validator import plan_validator
from obsidian_agent.nodes.executor import executor
from obsidian_agent.nodes.summarizer import summarizer
from obsidian_agent.nodes.error_handler import error_handler

logger = logging.getLogger(__name__)


def _make_safe(node_fn, node_name: str):
    """Wrap a node function to catch exceptions and write to error field."""

    def safe_node(state: AgentState) -> dict:
        try:
            return node_fn(state)
        except Exception as e:
            logger.error("[%s] Unhandled error: %s", node_name, e, exc_info=True)
            return {"error": f"[{node_name}] {type(e).__name__}: {e}"}

    safe_node.__name__ = node_name
    return safe_node


def build_graph(
    settings: Settings,
    obsidian_client: ObsidianClient,
) -> StateGraph:
    """Build and compile the LangGraph pipeline.

    Args:
        settings: Immutable configuration.
        obsidian_client: Shared HTTP client for mcpo proxy.

    Returns:
        Compiled StateGraph ready for .invoke().
    """
    llm = create_llm(settings)

    # Bind dependencies into node functions via closures
    def _intent_classifier(state: AgentState) -> dict:
        return intent_classifier(state, llm)

    def _web_search_extractor(state: AgentState) -> dict:
        return web_search_extractor(state)

    def _kb_retrieval_agent(state: AgentState) -> dict:
        return kb_retrieval_agent(state, obsidian_client)

    def _file_extraction_agent(state: AgentState) -> dict:
        return file_extraction_agent(state)

    def _image_description_agent(state: AgentState) -> dict:
        return image_description_agent(state, llm)

    def _context_aggregator(state: AgentState) -> dict:
        return context_aggregator(
            state, settings.context_token_budget, settings.chars_per_token
        )

    def _planner(state: AgentState) -> dict:
        return planner(state, llm)

    def _plan_validator(state: AgentState) -> dict:
        return plan_validator(state)

    def _executor(state: AgentState) -> dict:
        return executor(state, obsidian_client)

    def _summarizer(state: AgentState) -> dict:
        return summarizer(state, llm)

    def _error_handler(state: AgentState) -> dict:
        return error_handler(state)

    # Build the graph
    builder = StateGraph(AgentState)

    # Add all nodes (wrapped for safety)
    builder.add_node("intent_classifier", _make_safe(_intent_classifier, "intent_classifier"))
    builder.add_node("web_search_extractor", _make_safe(_web_search_extractor, "web_search_extractor"))
    builder.add_node("kb_retrieval_agent", _make_safe(_kb_retrieval_agent, "kb_retrieval_agent"))
    builder.add_node("file_extraction_agent", _make_safe(_file_extraction_agent, "file_extraction_agent"))
    builder.add_node("image_description_agent", _make_safe(_image_description_agent, "image_description_agent"))
    builder.add_node("context_aggregator", _make_safe(_context_aggregator, "context_aggregator"))
    builder.add_node("planner", _make_safe(_planner, "planner"))
    builder.add_node("plan_validator", _make_safe(_plan_validator, "plan_validator"))
    builder.add_node("executor", _make_safe(_executor, "executor"))
    builder.add_node("summarizer", _make_safe(_summarizer, "summarizer"))
    builder.add_node("error_handler", _make_safe(_error_handler, "error_handler"))

    # Entry edge
    builder.add_edge(START, "intent_classifier")

    # Fan-out: intent_classifier -> 4 context nodes
    # (These execute sequentially with sync nodes, but the graph structure
    #  supports parallel execution if nodes are made async later)
    builder.add_edge("intent_classifier", "web_search_extractor")
    builder.add_edge("intent_classifier", "kb_retrieval_agent")
    builder.add_edge("intent_classifier", "file_extraction_agent")
    builder.add_edge("intent_classifier", "image_description_agent")

    # Fan-in: all 4 context nodes -> context_aggregator
    builder.add_edge("web_search_extractor", "context_aggregator")
    builder.add_edge("kb_retrieval_agent", "context_aggregator")
    builder.add_edge("file_extraction_agent", "context_aggregator")
    builder.add_edge("image_description_agent", "context_aggregator")

    # context_aggregator -> planner
    builder.add_edge("context_aggregator", "planner")

    # planner -> plan_validator
    builder.add_edge("planner", "plan_validator")

    # Conditional: plan_validator -> executor | planner (retry) | error_handler
    max_retries = settings.max_plan_retries

    def route_after_validation(state: AgentState) -> str:
        if state.get("plan_valid", False):
            return "executor"
        if state.get("retry_count", 0) < max_retries:
            return "planner"
        return "error_handler"

    builder.add_conditional_edges(
        "plan_validator",
        route_after_validation,
        {"executor": "executor", "planner": "planner", "error_handler": "error_handler"},
    )

    # Conditional: executor -> summarizer | error_handler
    def route_after_execution(state: AgentState) -> str:
        if state.get("error", ""):
            return "error_handler"
        return "summarizer"

    builder.add_conditional_edges(
        "executor",
        route_after_execution,
        {"summarizer": "summarizer", "error_handler": "error_handler"},
    )

    # Terminal edges
    builder.add_edge("summarizer", END)
    builder.add_edge("error_handler", END)

    # Compile
    compiled = builder.compile()
    logger.info("LangGraph pipeline compiled successfully")

    return compiled
