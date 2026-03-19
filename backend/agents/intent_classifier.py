"""
Hybrid intent classification node for the LangGraph orchestrator.

Uses a two-pass approach:
  1. Regex-based fast pass — cheap and deterministic. If a strong match
     is found, it's used directly (confidence = high).
  2. LLM-based fallback — only called when regex is uncertain
     (confidence = medium). This handles natural/messy user input.

Supported intents:
  - rag: Ask a question from documents
  - appointment: Book an appointment
  - appointment_cancel: Cancel an in-progress booking
  - greeting: Say hello / small talk

Supports context switching: even mid-appointment, if the user clearly asks
a document question or wants to cancel, the intent is reclassified so the
correct agent handles the message.
"""

from __future__ import annotations

import re

from backend.agents.state import AgentState
from backend.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Regex patterns (case-insensitive) — ordered by specificity
# ---------------------------------------------------------------------------

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
    r"\b(know\s+more|learn\s+more|more\s+about|details?\s+about|information\s+(about|on))\b",
    r"\?$",
]

_EXIT_APPOINTMENT_PATTERNS = [
    r"\b(cancel|stop|quit|exit|nevermind|never\s*mind|abort)\b",
    r"\b(go\s*back|back\s*to)\b",
]

_LOOKS_LIKE_QUESTION_PATTERNS = [
    r"\b(what|how|why|when|where|who|explain|describe|tell me about)\b.*\b(your|the|this|a|an)\b",
    r"^(what|how|why|when|where|who)\s.{15,}",
    r"\?$",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _matches_any(text: str, patterns: list[str]) -> bool:
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)


def _count_pattern_matches(text: str, patterns: list[str]) -> int:
    """Count how many patterns match — more matches = higher certainty."""
    return sum(1 for p in patterns if re.search(p, text, re.IGNORECASE))


def _is_plausible_field_input(text: str, step: str) -> bool:
    """
    Check whether *text* looks like a plausible value for the current
    appointment step (name, phone, email, date, confirm).
    """
    if step == "name":
        return len(text) < 60 and "?" not in text
    if step == "phone":
        digits = sum(c.isdigit() for c in text)
        return digits >= 6
    if step == "email":
        return "@" in text
    if step == "date":
        return len(text) < 80 and "?" not in text
    if step == "confirm":
        confirm_deny_pattern = re.compile(
            r"\b(yes|y|no|nah|confirm|cancel|sure|ok|okay|nope|yep|yeah"
            r"|absolutely|go\s+ahead|please|don't|stop|nevermind)\b",
            re.IGNORECASE,
        )
        return bool(confirm_deny_pattern.search(text))
    return True


def _regex_classify(message: str) -> tuple[str | None, str]:
    """
    First pass: attempt to classify with regex.

    Returns:
        (intent, confidence) where confidence is "high" or "low".
        intent is None if regex couldn't determine with any confidence.
    """
    appointment_hits = _count_pattern_matches(message, _APPOINTMENT_PATTERNS)
    greeting_hits = _count_pattern_matches(message, _GREETING_PATTERNS)
    rag_hits = _count_pattern_matches(message, _RAG_PATTERNS)

    # Strong match: multiple patterns hit, or an appointment keyword is clear
    if appointment_hits >= 1 and rag_hits == 0:
        return "appointment", "high"

    if greeting_hits >= 1 and appointment_hits == 0 and rag_hits == 0:
        # Only classify as greeting if the message is short (actual greeting)
        if len(message.split()) <= 5:
            return "greeting", "high"

    if rag_hits >= 2 and appointment_hits == 0:
        return "rag", "high"

    # Single rag hit with a question mark — fairly confident
    if rag_hits == 1 and "?" in message and appointment_hits == 0:
        return "rag", "high"

    # Ambiguous: appointment and rag patterns both match
    if appointment_hits >= 1 and rag_hits >= 1:
        # Check if the user is asking *about* a topic (RAG) vs requesting
        # to *perform* a booking action.
        # Phrases like "know more about", "tell me about", "what is",
        # "explain", "learn about" indicate information-seeking, not booking.
        info_seeking_pattern = re.compile(
            r"\b(know\s+more|tell\s+me|learn|explain|describe|what\s+is|what\s+are"
            r"|how\s+does|how\s+do|details?\s+about|information\s+(about|on))\b",
            re.IGNORECASE,
        )
        if info_seeking_pattern.search(message):
            return "rag", "high"
        # e.g. "Can you book an appointment?" — appointment takes priority
        return "appointment", "high"

    # Single weak match
    if rag_hits == 1:
        return "rag", "low"

    # No patterns matched at all
    return None, "low"


# ---------------------------------------------------------------------------
# Main classification node
# ---------------------------------------------------------------------------

def classify_intent(state: AgentState) -> AgentState:
    """
    LangGraph node: inspect ``user_message`` and set ``intent``.

    Two-pass hybrid approach:
      1. Regex first — fast, deterministic, high confidence
      2. LLM fallback — only when regex is uncertain

    When an appointment flow is in progress, the classifier examines
    message content for context switching. If it looks like a document
    question or exit command, the intent changes accordingly.
    """
    message = state.get("user_message", "").strip()
    current_step = state.get("appointment_step", "")
    llm_model = state.get("llm_model", "")

    if not message:
        state["intent"] = "unknown"
        state["intent_confidence"] = "high"
        state["intent_source"] = "regex"
        return state

    # --- Mid-appointment context switching --------------------------------
    if current_step and current_step != "complete":
        # Check if the user wants to exit
        if _matches_any(message, _EXIT_APPOINTMENT_PATTERNS):
            logger.info(
                "User requested exit from appointment flow (step=%s)",
                current_step,
            )
            state["intent"] = "appointment_cancel"
            state["intent_confidence"] = "high"
            state["intent_source"] = "regex"
            return state

        # Check if message looks like a question rather than a field value
        if (
            _matches_any(message, _LOOKS_LIKE_QUESTION_PATTERNS)
            and not _is_plausible_field_input(message, current_step)
        ):
            logger.info(
                "Detected context switch from appointment (step=%s) to RAG",
                current_step,
            )
            state["intent"] = "rag"
            state["intent_confidence"] = "high"
            state["intent_source"] = "regex"
            return state

        # Otherwise continue the appointment flow
        logger.debug("Appointment flow in progress (step=%s), keeping intent", current_step)
        state["intent"] = "appointment"
        state["intent_confidence"] = "high"
        state["intent_source"] = "context"
        return state

    # --- Pass 1: Regex classification -------------------------------------
    regex_intent, regex_confidence = _regex_classify(message)

    if regex_intent and regex_confidence == "high":
        state["intent"] = regex_intent
        state["intent_confidence"] = "high"
        state["intent_source"] = "regex"
        logger.info(
            "Regex classified intent as '%s' (confidence=high) for: %.60s",
            regex_intent, message,
        )
        return state

    # --- Pass 2: LLM fallback (regex uncertain or no match) ---------------
    logger.info(
        "Regex uncertain (intent=%s, confidence=%s), falling back to LLM for: %.60s",
        regex_intent, regex_confidence, message,
    )

    try:
        from backend.services.llm_provider import classify_intent_with_llm

        llm_intent = classify_intent_with_llm(message, model=llm_model)

        if llm_intent:
            state["intent"] = llm_intent
            state["intent_confidence"] = "medium"
            state["intent_source"] = "llm"
            logger.info(
                "LLM classified intent as '%s' (confidence=medium) for: %.60s",
                llm_intent, message,
            )
            return state
    except Exception as exc:
        logger.warning("LLM classification failed, using regex fallback: %s", exc)

    # --- Final fallback: use regex result or default to rag ---------------
    final_intent = regex_intent or "rag"
    state["intent"] = final_intent
    state["intent_confidence"] = "low"
    state["intent_source"] = "fallback"
    logger.info(
        "Fallback classified intent as '%s' (confidence=low) for: %.60s",
        final_intent, message,
    )
    return state
