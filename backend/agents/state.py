"""
Shared agent state definition for LangGraph.

All agents operate on the same ``AgentState`` TypedDict so they can be
composed into a single graph.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from typing_extensions import TypedDict


class AgentState(TypedDict, total=False):
    """
    Shared state passed between LangGraph nodes.

    Attributes:
        user_message: The current user input.
        session_id: Conversation session identifier.
        chat_history: Prior messages in this session.
        intent: Detected intent (rag | appointment | greeting | appointment_cancel).
        intent_confidence: How confident the classifier is (high | medium | low).
        intent_source: Which classifier determined the intent (regex | llm).
        rag_context: Assembled document context for LLM.
        rag_sources: Source document names used.
        appointment_data: Partially or fully collected appointment fields.
        appointment_step: Current step in the booking flow.
        response: Final response text to return to the user.
        error: Error message, if any.
    """

    user_message: str
    session_id: str
    chat_history: List[Dict[str, str]]
    intent: str
    intent_confidence: str
    intent_source: str
    llm_model: str
    rag_context: str
    rag_sources: List[str]
    appointment_data: Dict[str, Any]
    appointment_step: str
    response: str
    error: Optional[str]
