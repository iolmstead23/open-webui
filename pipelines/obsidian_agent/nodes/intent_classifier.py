"""Intent classifier node -- classifies user message into IntentOutput.

Dependencies: llm, state.
Owns: intent field.
"""
import logging

from langchain_core.messages import SystemMessage, HumanMessage

from obsidian_agent.state.schema import AgentState, IntentOutput
from obsidian_agent.llm.client import invoke_structured

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an intent classifier for an Obsidian vault assistant.
Classify the user's message into exactly one of these intents:

- vault_read: User wants to retrieve, find, or search vault content. Also use this
  when the user asks about a topic, concept, or subject area where they may have notes
  (e.g. "what is X", "explain X", "show me X", "tell me about X", "review X").
- vault_write: User wants to create, modify, append, or update vault content.
- vault_delete: User wants to delete or remove vault content. Only use this if the user
  explicitly mentions deleting or removing.
- general_qa: ONLY use this when the user is asking a question that clearly does NOT
  require personal notes (e.g. universal math facts, calculations, off-topic requests).
  When in doubt between vault_read and general_qa, prefer vault_read.
- periodic_note: User is specifically referencing a periodic note (daily, weekly, monthly,
  quarterly, yearly).
- recent_changes: User wants to know what changed recently in the vault.

Think step by step about the user's intent, then provide your classification.
Respond with a JSON object containing "intent" and "reasoning" fields."""


def intent_classifier(state: AgentState, llm) -> dict:
    """Classify user message intent.

    Args:
        state: Current pipeline state.
        llm: ChatOllama instance.

    Returns:
        Partial state dict with intent field.
    """
    user_message = state.get("user_message", "")
    conversation_history = state.get("conversation_history", [])

    messages = [SystemMessage(content=SYSTEM_PROMPT)]

    # Add recent conversation context (last 4 messages = up to 2 user/assistant exchanges)
    for turn in conversation_history[-4:]:
        role = turn.get("role", "user")
        content = turn.get("content", "")
        if role == "user":
            messages.append(HumanMessage(content=content))
        else:
            from langchain_core.messages import AIMessage
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
            intent="vault_read",
            reasoning=f"Fallback due to classification error: {e}",
        )
        return {
            "intent": fallback.model_dump(),
            "error": f"[intent_classifier] Classification failed, defaulting to general_qa: {e}",
        }
