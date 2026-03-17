"""
Greeting handler node for LangGraph.

Returns a friendly response when the user greets the agent.
"""

from __future__ import annotations

from backend.agents.state import AgentState
from backend.logger import get_logger

logger = get_logger(__name__)


def handle_greeting(state: AgentState) -> AgentState:
    """
    LangGraph node: respond to greetings and light conversation.
    """
    message = state.get("user_message", "").strip().lower()
    logger.debug("Greeting handler received: '%s'", message)

    state["response"] = (
        "Hello! I'm your assistant. I can help you with:\n"
        "- **Answering questions** from our documents\n"
        "- **Booking appointments**\n\n"
        "How can I help you today?"
    )
    return state
