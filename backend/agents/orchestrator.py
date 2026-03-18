"""
LangGraph orchestrator — the main graph that routes user messages
to the appropriate agent based on detected intent.

Graph flow:
    classify_intent -> (conditional edge) -> rag_agent | appointment_agent | greeting_handler
"""

from __future__ import annotations

import uuid
from typing import Dict, List

from langgraph.graph import END, StateGraph

from backend.agents.appointment_agent import handle_appointment
from backend.agents.greeting_handler import handle_greeting
from backend.agents.intent_classifier import classify_intent
from backend.agents.rag_agent import handle_rag_query
from backend.agents.state import AgentState
from backend.config import GEMINI_MODEL_NAME
from backend.logger import get_logger
from backend.services.guardrails_service import check_message
from backend.services.persistence_service import (
    load_all_conversations,
    save_conversation,
)

logger = get_logger(__name__)


def _handle_appointment_cancel(state: AgentState) -> AgentState:
    """LangGraph node: cancel the in-progress appointment and reset state."""
    logger.info("Cancelling appointment flow (was at step=%s)", state.get("appointment_step"))
    state["appointment_step"] = "complete"
    state["appointment_data"] = {}
    state["response"] = (
        "No problem — I've cancelled the appointment booking. "
        "Feel free to ask me anything else or start a new booking whenever you're ready!"
    )
    return state

# ---------------------------------------------------------------------------
# Session stores — loaded from disk on startup, flushed after each message
# ---------------------------------------------------------------------------
_sessions: Dict[str, List[Dict[str, str]]] = load_all_conversations()
_session_appointment_state: Dict[str, dict] = {}


def clear_session(session_id: str) -> bool:
    """
    Clear all state for a session (chat history + appointment state).

    Returns True if the session existed, False if not found.
    """
    found = session_id in _sessions or session_id in _session_appointment_state
    _sessions.pop(session_id, None)
    _session_appointment_state.pop(session_id, None)

    # Remove persisted conversation file
    from backend.services.persistence_service import delete_conversation
    delete_conversation(session_id)

    if found:
        logger.info("Session cleared: %s", session_id)
    else:
        logger.info("Session not found for clearing: %s", session_id)
    return found


def _route_by_intent(state: AgentState) -> str:
    """Conditional edge: return the node name matching the classified intent."""
    intent = state.get("intent", "unknown")
    route_map = {
        "rag": "rag_agent",
        "appointment": "appointment_agent",
        "appointment_cancel": "appointment_cancel_handler",
        "greeting": "greeting_handler",
    }
    destination = route_map.get(intent, "rag_agent")
    logger.debug("Routing to '%s' (intent=%s)", destination, intent)
    return destination


def _build_graph() -> StateGraph:
    """Construct the LangGraph state graph (built once, reused per request)."""
    graph = StateGraph(AgentState)

    # Nodes
    graph.add_node("classify_intent", classify_intent)
    graph.add_node("rag_agent", handle_rag_query)
    graph.add_node("appointment_agent", handle_appointment)
    graph.add_node("appointment_cancel_handler", _handle_appointment_cancel)
    graph.add_node("greeting_handler", handle_greeting)

    # Edges
    graph.set_entry_point("classify_intent")
    graph.add_conditional_edges(
        "classify_intent",
        _route_by_intent,
        {
            "rag_agent": "rag_agent",
            "appointment_agent": "appointment_agent",
            "appointment_cancel_handler": "appointment_cancel_handler",
            "greeting_handler": "greeting_handler",
        },
    )
    graph.add_edge("rag_agent", END)
    graph.add_edge("appointment_agent", END)
    graph.add_edge("appointment_cancel_handler", END)
    graph.add_edge("greeting_handler", END)

    return graph


# Compile once at module level
_compiled_graph = _build_graph().compile()


def process_message(
    user_message: str,
    session_id: str | None = None,
    llm_model: str | None = None,
) -> dict:
    """
    Public entry point: process a user message through the agent graph.

    Args:
        user_message: The raw text from the user.
        session_id: Optional session identifier. A new one is created if absent.
        llm_model: Optional Gemini model override for this request.

    Returns:
        Dict with ``reply``, ``session_id``, ``intent``, ``sources``, and ``model_used``.
    """
    if not session_id:
        session_id = uuid.uuid4().hex
        logger.info("New session created: %s", session_id)

    # Retrieve or initialise session state
    chat_history = _sessions.setdefault(session_id, [])
    appt_state = _session_appointment_state.setdefault(session_id, {})

    resolved_model = llm_model or GEMINI_MODEL_NAME

    initial_state: AgentState = {
        "user_message": user_message,
        "session_id": session_id,
        "chat_history": chat_history,
        "intent": "",
        "intent_confidence": "",
        "intent_source": "",
        "llm_model": resolved_model,
        "rag_context": "",
        "rag_sources": [],
        "appointment_data": appt_state.get("data", {}),
        "appointment_step": appt_state.get("step", ""),
        "response": "",
        "error": None,
    }

    logger.info(
        "Processing message (session=%s): %.80s",
        session_id,
        user_message,
    )

    # --- Guardrails (skip when mid-appointment to avoid blocking field inputs) ---
    active_appointment = appt_state.get("step", "") not in ("", "complete")
    if not active_appointment:
        guardrail_result = check_message(user_message)
        if guardrail_result.is_blocked:
            logger.info(
                "Guardrail blocked message (reason=%s, session=%s): %.60s",
                guardrail_result.reason,
                session_id,
                user_message,
            )
            # Still save to chat history so the user sees it in context
            chat_history.append({"role": "user", "content": user_message})
            chat_history.append({"role": "assistant", "content": guardrail_result.suggested_response})
            save_conversation(session_id, chat_history)
            return {
                "reply": guardrail_result.suggested_response,
                "session_id": session_id,
                "intent": f"blocked:{guardrail_result.reason}",
                "intent_confidence": "high",
                "intent_source": "guardrail",
                "sources": [],
                "model_used": resolved_model,
            }

    final_state = _compiled_graph.invoke(initial_state)

    # Persist conversational context
    chat_history.append({"role": "user", "content": user_message})
    chat_history.append({"role": "assistant", "content": final_state.get("response", "")})

    # Keep history bounded (last 20 messages)
    if len(chat_history) > 20:
        _sessions[session_id] = chat_history[-20:]
        chat_history = _sessions[session_id]

    # Flush conversation to disk
    save_conversation(session_id, chat_history)

    # Persist appointment flow state
    _session_appointment_state[session_id] = {
        "data": final_state.get("appointment_data", {}),
        "step": final_state.get("appointment_step", ""),
    }

    result = {
        "reply": final_state.get("response", "I'm not sure how to help with that."),
        "session_id": session_id,
        "intent": final_state.get("intent", "unknown"),
        "intent_confidence": final_state.get("intent_confidence", ""),
        "intent_source": final_state.get("intent_source", ""),
        "sources": final_state.get("rag_sources", []),
        "model_used": resolved_model,
    }

    logger.info(
        "Response generated (session=%s, intent=%s, len=%d)",
        session_id,
        result["intent"],
        len(result["reply"]),
    )
    return result
