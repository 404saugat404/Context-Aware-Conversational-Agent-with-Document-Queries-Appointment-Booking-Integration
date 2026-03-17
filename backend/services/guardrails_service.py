"""
Input guardrails for filtering harmful, off-topic, or nonsensical messages.

These checks run BEFORE the message reaches any agent, providing a fast
rejection path that doesn't consume LLM tokens.
"""

from __future__ import annotations

import re

from backend.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Blocked content patterns
# ---------------------------------------------------------------------------

_HARMFUL_PATTERNS = [
    r"\b(hack|exploit|crack\s+password|sql\s*injection|xss|ddos)\b",
    r"\b(how\s+to\s+(?:kill|hurt|harm|attack|steal|rob))\b",
    r"\b(make\s+(?:a\s+)?(?:bomb|weapon|drug|poison))\b",
    r"\b(illegal|illicit)\s+(?:activity|drug|substance)\b",
]

_PERSONAL_DATA_REQUEST_PATTERNS = [
    r"\b(give\s+me|share|tell\s+me)\s+(?:your|the)\s+(?:password|api\s*key|secret|credentials?|token)\b",
    r"\b(credit\s*card|ssn|social\s+security)\s*(?:number)?\b",
]

_PROMPT_INJECTION_PATTERNS = [
    r"(?:ignore|forget|disregard)\s+(?:all\s+)?(?:previous|prior|above)\s+(?:instructions?|prompts?|rules?)",
    r"you\s+are\s+now\s+(?:a|an)\s+(?:different|new)",
    r"(?:system|admin)\s*:\s*",
    r"act\s+as\s+(?:if\s+)?(?:you\s+are|you're)\s+(?:a|an)",
    r"\[(?:system|INST)\]",
]

_GIBBERISH_PATTERN = re.compile(
    r"^[^a-zA-Z]*$"  # no letters at all
    r"|^(.)\1{5,}$",  # same character repeated 6+ times
    re.IGNORECASE,
)

# Topics clearly outside our scope (not document Q&A or appointment booking)
_OFF_TOPIC_PATTERNS = [
    r"\b(write\s+(?:me\s+)?(?:a|an)\s+(?:\w+\s+)?(?:essay|poem|story|song|code|script|program))\b",
    r"\b(translate|convert)\s+(?:this|the\s+following)\s+(?:to|into)\b",
    r"\b((?:play|tell)\s+(?:me\s+)?(?:a\s+)?(?:game|joke|riddle))\b",
    r"\b(what(?:'s| is)\s+(?:the\s+)?(?:weather|time|date|news))\b",
    r"\b((?:who|what)\s+(?:is|are|was|were)\s+(?:the\s+)?(?:president|capital|tallest|largest|fastest))\b",
    r"\b(calculate|solve|compute|math)\b.*\b(\d+\s*[\+\-\*\/\^]\s*\d+)\b",
    r"\b(recipe|cook|bake|ingredient)\b",
    r"\b(stock\s+price|crypto|bitcoin|trading)\b",
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class GuardrailResult:
    """Result of a guardrail check."""

    __slots__ = ("is_blocked", "reason", "suggested_response")

    def __init__(
        self,
        is_blocked: bool = False,
        reason: str = "",
        suggested_response: str = "",
    ):
        self.is_blocked = is_blocked
        self.reason = reason
        self.suggested_response = suggested_response


_SCOPE_RESPONSE = (
    "I'm sorry, that's outside what I can help with. I'm designed to:\n"
    "- **Answer questions** based on our documents\n"
    "- **Book appointments** for you\n\n"
    "Is there anything I can help you with in those areas?"
)

_HARMFUL_RESPONSE = (
    "I'm not able to help with that kind of request. "
    "I'm here to answer questions from our documents and help book appointments. "
    "Please let me know if I can assist with either of those!"
)

_INJECTION_RESPONSE = (
    "I detected something unusual in your message. "
    "Could you please rephrase your question? "
    "I'm here to help with document queries and appointment booking."
)

_GIBBERISH_RESPONSE = (
    "I couldn't understand that message. Could you please rephrase? "
    "I can help you with document questions or booking an appointment."
)


def check_message(message: str) -> GuardrailResult:
    """
    Run all guardrail checks on a user message.

    Returns a ``GuardrailResult``. If ``is_blocked`` is True, the message
    should not be forwarded to any agent — use ``suggested_response`` instead.
    """
    text = message.strip()

    if not text:
        return GuardrailResult(
            is_blocked=True,
            reason="empty_message",
            suggested_response=_GIBBERISH_RESPONSE,
        )

    # Too short to be meaningful (single character, etc.)
    if len(text) < 2:
        return GuardrailResult(
            is_blocked=True,
            reason="too_short",
            suggested_response=_GIBBERISH_RESPONSE,
        )

    # Gibberish detection
    if _GIBBERISH_PATTERN.match(text):
        logger.info("Guardrail blocked gibberish: %.40s", text)
        return GuardrailResult(
            is_blocked=True,
            reason="gibberish",
            suggested_response=_GIBBERISH_RESPONSE,
        )

    lowered = text.lower()

    # Prompt injection attempts
    for pattern in _PROMPT_INJECTION_PATTERNS:
        if re.search(pattern, lowered, re.IGNORECASE):
            logger.warning("Guardrail blocked prompt injection attempt: %.60s", text)
            return GuardrailResult(
                is_blocked=True,
                reason="prompt_injection",
                suggested_response=_INJECTION_RESPONSE,
            )

    # Harmful content
    for pattern in _HARMFUL_PATTERNS:
        if re.search(pattern, lowered, re.IGNORECASE):
            logger.warning("Guardrail blocked harmful content: %.60s", text)
            return GuardrailResult(
                is_blocked=True,
                reason="harmful_content",
                suggested_response=_HARMFUL_RESPONSE,
            )

    # Personal data requests
    for pattern in _PERSONAL_DATA_REQUEST_PATTERNS:
        if re.search(pattern, lowered, re.IGNORECASE):
            logger.warning("Guardrail blocked personal data request: %.60s", text)
            return GuardrailResult(
                is_blocked=True,
                reason="personal_data_request",
                suggested_response=_HARMFUL_RESPONSE,
            )

    # Off-topic detection — only when NOT in an active appointment flow
    # (the appointment step check happens in the orchestrator)
    for pattern in _OFF_TOPIC_PATTERNS:
        if re.search(pattern, lowered, re.IGNORECASE):
            logger.info("Guardrail flagged off-topic message: %.60s", text)
            return GuardrailResult(
                is_blocked=True,
                reason="off_topic",
                suggested_response=_SCOPE_RESPONSE,
            )

    # Message too long (possible abuse)
    if len(text) > 2000:
        logger.info("Guardrail blocked oversized message: %d chars", len(text))
        return GuardrailResult(
            is_blocked=True,
            reason="too_long",
            suggested_response="Your message is too long. Please keep it under 2000 characters.",
        )

    return GuardrailResult(is_blocked=False)
