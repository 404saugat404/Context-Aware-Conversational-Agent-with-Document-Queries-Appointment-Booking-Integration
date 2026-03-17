"""
Intent classification node for the LangGraph orchestrator.

Determines whether the user wants to:
  - Ask a question (RAG)
  - Book an appointment
  - Exchange a greeting
  - Something else (fallback)

Supports context switching: even mid-appointment, if the user clearly asks
a document question or wants to cancel, the intent is reclassified so the
correct agent handles the message. The appointment flow is paused (not lost)
and resumes when the user comes back to it.
"""

from __future__ import annotations

import re

from backend.agents.state import AgentState
from backend.logger import get_logger

logger = get_logger(__name__)

# Keyword patterns (case-insensitive) — ordered by specificity
_APPOINTMENT_PATTERNS = [
    r"\b(book|schedule|appointment|booking|reserve)\b",
    r"\b(set up|arrange)\s+(a\s+)?(meeting|visit|consultation)\b",
    r"\b(continue|resume)\s+(booking|appointment)\b",
]

_GREETING_PATTERNS = [
    r"^(hi|hello|hey|good\s+(morning|afternoon|evening)|howdy|greetings)\b",
]

_RAG_PATTERNS = [
    r"\b(what|how|why|when|where|who|explain|describe|tell me|can you)\b",
    r"\?$",
]

# Patterns that signal the user wants to leave the appointment flow
_EXIT_APPOINTMENT_PATTERNS = [
    r"\b(cancel|stop|quit|exit|nevermind|never\s*mind|abort)\b",
    r"\b(go\s*back|back\s*to)\b",
]

# Patterns that look like a document/knowledge question rather than
# a short field value (name, phone, email, date)
_LOOKS_LIKE_QUESTION_PATTERNS = [
    r"\b(what|how|why|when|where|who|explain|describe|tell me about)\b.*\b(your|the|this|a|an)\b",
    r"^(what|how|why|when|where|who)\s.{15,}",  # question word + long sentence
    r"\?$",
]


def _matches_any(text: str, patterns: list[str]) -> bool:
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)


def _is_plausible_field_input(text: str, step: str) -> bool:
    """
    Check whether *text* looks like a plausible value for the current
    appointment step. This helps distinguish between "John Smith" (a name)
    and "What services do you offer?" (a RAG question) when the step is 'name'.
    """
    if step == "name":
        # Names are typically short and don't contain question marks
        return len(text) < 60 and "?" not in text
    if step == "phone":
        # Phone input is mostly digits
        digits = sum(c.isdigit() for c in text)
        return digits >= 6
    if step == "email":
        return "@" in text
    if step == "date":
        # Date inputs are usually short
        return len(text) < 80 and "?" not in text
    if step == "confirm":
        confirm_deny_pattern = re.compile(r"\b(yes|y|no|nah|confirm|cancel|sure|ok|okay|nope|yep|yeah|absolutely|go\s+ahead|please|don't|stop|nevermind)\b", re.IGNORECASE)
        return bool(confirm_deny_pattern.search(text))
    return True


def classify_intent(state: AgentState) -> AgentState:
    """
    LangGraph node: inspect ``user_message`` and set ``intent``.

    When an appointment flow is in progress, the classifier still examines
    the message content. If it looks like a document question or an exit
    command, the intent is changed accordingly so the user can seamlessly
    switch contexts. The appointment state is preserved (paused) and can
    be resumed later.
    """
    message = state.get("user_message", "").strip()
    current_step = state.get("appointment_step", "")

    if not message:
        state["intent"] = "unknown"
        return state

    # --- Mid-appointment context switching --------------------------------
    if current_step and current_step != "complete":
        # Check if the user wants to exit the appointment flow
        if _matches_any(message, _EXIT_APPOINTMENT_PATTERNS):
            logger.info(
                "User requested exit from appointment flow (step=%s)",
                current_step,
            )
            state["intent"] = "appointment_cancel"
            return state

        # Check if the message looks like a question rather than a field value
        if (
            _matches_any(message, _LOOKS_LIKE_QUESTION_PATTERNS)
            and not _is_plausible_field_input(message, current_step)
        ):
            logger.info(
                "Detected context switch from appointment (step=%s) to RAG",
                current_step,
            )
            state["intent"] = "rag"
            return state

        # Otherwise continue the appointment flow
        logger.debug("Appointment flow in progress (step=%s), keeping intent", current_step)
        state["intent"] = "appointment"
        return state

    # --- Normal classification (no active appointment flow) ---------------
    if _matches_any(message, _APPOINTMENT_PATTERNS):
        intent = "appointment"
    elif _matches_any(message, _GREETING_PATTERNS):
        intent = "greeting"
    elif _matches_any(message, _RAG_PATTERNS):
        intent = "rag"
    else:
        # Default to RAG for any substantive question
        intent = "rag"

    state["intent"] = intent
    logger.info("Classified intent as '%s' for message: %.60s", intent, message)
    return state
