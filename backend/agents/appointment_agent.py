"""
Appointment booking agent node for LangGraph.

Drives a multi-step conversational form collecting: name, phone, email, date.

Hybrid extraction approach:
  1. Regex-based extraction (fast, deterministic)
  2. Pattern search (scan raw input for phone digits / email patterns)
  3. LLM-based fallback when regex can't cleanly extract the value

Smart features:
  - Multi-field extraction: parses multiple fields from a single message
  - Step skipping: only asks for fields that are still missing
  - Never overwrites valid existing data
"""

from __future__ import annotations

import re
from datetime import date
from typing import Optional

from pydantic import ValidationError

from backend.agents.state import AgentState
from backend.logger import get_logger
from backend.models.appointment_schema import AppointmentRequest
from backend.services.appointment_service import book_appointment, parse_and_validate_date
from backend.services.tools_service import validate_email, validate_phone_number

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Regex patterns for field extraction
# ---------------------------------------------------------------------------

_NAME_PREFIX_PATTERN = re.compile(
    r"^(?:(?:hi|hello|hey|greetings)[,.\s!]*)?(?:my\s+name\s+is|i\s+am|i'm|it's|its|this\s+is|call\s+me)\s+",
    re.IGNORECASE,
)

_PHONE_PREFIX_PATTERN = re.compile(
    r"^(?:(?:my\s+)?(?:phone\s+)?(?:number|phone|cell|mobile)(?:\s+is)?|it's|its|phone\s*(?:number)?)[:\s]+",
    re.IGNORECASE,
)

_EMAIL_PREFIX_PATTERN = re.compile(
    r"^(?:(?:my\s+)?(?:email\s+)?(?:address|email|mail|id)(?:\s+is)?|it's|its|email\s*(?:address)?)[:\s]+",
    re.IGNORECASE,
)

_INLINE_NAME_PATTERN = re.compile(
    r"(?:my\s+name\s+is|i\s+am|i'm|call\s+me)\s+([A-Za-z][A-Za-z\s\-]{0,50}?)(?:\s*[,.]|\s+(?:my|and|phone|email|number|cell|mobile|date|for)\b|$)",
    re.IGNORECASE,
)

_PHONE_DIGIT_PATTERN = re.compile(r"\+?\d[\d\s\-]{8,}")
_EMAIL_SCAN_PATTERN = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")

_CONFIRM_PATTERN = re.compile(
    r"\b(yes|y|confirm|sure|ok|okay|yep|yeah|absolutely|go\s+ahead|please)\b",
    re.IGNORECASE,
)
_DENY_PATTERN = re.compile(
    r"\b(no|nah|cancel|nope|don't|stop|nevermind|never\s*mind)\b",
    re.IGNORECASE,
)

# Required fields in collection order
_REQUIRED_FIELDS = ["name", "phone", "email", "preferred_date"]


# ---------------------------------------------------------------------------
# Regex-based single-field extractors
# ---------------------------------------------------------------------------

# Words that are clearly intent keywords, not names
_NOT_A_NAME_WORDS = {
    "book", "schedule", "appointment", "booking", "reserve", "cancel",
    "hi", "hello", "hey", "help", "want", "need", "would", "like",
    "please", "thanks", "thank", "you", "can", "could", "set", "up",
    "arrange", "meeting", "visit", "consultation", "continue", "resume",
    "a", "an", "the", "to", "for", "me", "my", "i", "do", "make",
    "get", "with", "have", "it", "this", "that", "or", "and", "is",
}


def _extract_name_regex(raw_input: str) -> Optional[str]:
    """Try to extract a name using regex prefix stripping."""
    cleaned = _NAME_PREFIX_PATTERN.sub("", raw_input.strip())
    cleaned = cleaned.rstrip(".!,").strip()
    if not cleaned:
        return None

    # Must be only letters/spaces/hyphens
    if not all(c.isalpha() or c.isspace() or c == '-' for c in cleaned):
        return None

    # Reject if all words are common intent/stop words (not a real name)
    words = {w.lower() for w in cleaned.split()}
    if words and words.issubset(_NOT_A_NAME_WORDS):
        return None

    # Names should have at most ~4 words
    if len(cleaned.split()) > 5:
        return None

    return cleaned.title()


def _extract_phone_regex(raw_input: str) -> Optional[str]:
    """Try to extract a phone number using regex."""
    # First try prefix stripping
    stripped = _PHONE_PREFIX_PATTERN.sub("", raw_input.strip())
    stripped = stripped.replace(" ", "").replace("-", "").rstrip(".!,").strip()
    if validate_phone_number(stripped):
        return stripped

    # Scan for digit patterns anywhere in the input
    match = _PHONE_DIGIT_PATTERN.search(raw_input)
    if match:
        found = match.group().replace(" ", "").replace("-", "")
        if validate_phone_number(found):
            return found
    return None


def _extract_email_regex(raw_input: str) -> Optional[str]:
    """Try to extract an email using regex."""
    # First try prefix stripping
    stripped = _EMAIL_PREFIX_PATTERN.sub("", raw_input.strip())
    stripped = stripped.rstrip(".!,").strip().lower()
    if validate_email(stripped):
        return stripped

    # Scan for email pattern anywhere in the input
    match = _EMAIL_SCAN_PATTERN.search(raw_input)
    if match:
        return match.group().lower()
    return None


def _extract_date_regex(raw_input: str) -> Optional[date]:
    """Try to parse date from text using dateparser/parsedatetime."""
    return parse_and_validate_date(raw_input)


# ---------------------------------------------------------------------------
# Multi-field regex scan
# ---------------------------------------------------------------------------

def _extract_fields_regex(message: str) -> dict[str, Optional[str]]:
    """
    Scan a message and extract as many appointment fields as possible via regex.

    Returns dict with keys: name, phone, email, preferred_date.
    Values are extracted strings or None.
    """
    result: dict[str, Optional[str]] = {
        "name": None, "phone": None, "email": None, "preferred_date": None,
    }

    # Phone: most reliable regex extraction
    phone = _extract_phone_regex(message)
    if phone:
        result["phone"] = phone

    # Email: also reliable
    email = _extract_email_regex(message)
    if email:
        result["email"] = email

    # Name: try inline pattern first, then direct regex extraction
    name_match = _INLINE_NAME_PATTERN.search(message)
    if name_match:
        name = name_match.group(1).strip().rstrip(".!,")
        if len(name) >= 2 and any(c.isalpha() for c in name):
            result["name"] = name.title()
    if not result["name"]:
        # Try direct extraction (strip prefix and validate)
        name_direct = _extract_name_regex(message)
        if name_direct:
            result["name"] = name_direct

    # Date: only try parsing if the message contains date-like keywords
    # to avoid dateparser aggressively interpreting names/random text as dates
    _DATE_HINT_PATTERN = re.compile(
        r"\b(january|february|march|april|may|june|july|august|september|october|november|december"
        r"|jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec"
        r"|monday|tuesday|wednesday|thursday|friday|saturday|sunday"
        r"|tomorrow|today|next\s+\w+|coming\s+\w+"
        r"|\d{1,2}[/-]\d{1,2}|\d{4}[/-]\d{1,2}[/-]\d{1,2}"
        r"|\d{1,2}(?:st|nd|rd|th))\b",
        re.IGNORECASE,
    )
    if _DATE_HINT_PATTERN.search(message):
        parsed_date = _extract_date_regex(message)
        if parsed_date:
            result["preferred_date"] = parsed_date.isoformat()

    return result


# ---------------------------------------------------------------------------
# Hybrid extraction: regex first, LLM fallback
# ---------------------------------------------------------------------------

def _extract_single_field(
    raw_input: str, field_type: str, model: str = ""
) -> Optional[str]:
    """
    Extract a single field value using regex first, then LLM fallback.

    Args:
        raw_input: The raw user message.
        field_type: One of "name", "phone", "email", "preferred_date".
        model: LLM model for fallback.

    Returns:
        Extracted value or None.
    """
    # Pass 1: Regex
    if field_type == "name":
        result = _extract_name_regex(raw_input)
        if result:
            return result
    elif field_type == "phone":
        result = _extract_phone_regex(raw_input)
        if result:
            return result
    elif field_type == "email":
        result = _extract_email_regex(raw_input)
        if result:
            return result
    elif field_type == "preferred_date":
        result = _extract_date_regex(raw_input)
        if result:
            return result.isoformat()

    # Pass 2: LLM fallback
    llm_field_type = "date" if field_type == "preferred_date" else field_type
    logger.info("Regex failed for %s, trying LLM on: %.60s", field_type, raw_input)
    try:
        from backend.services.llm_provider import extract_field_with_llm
        llm_result = extract_field_with_llm(raw_input, llm_field_type, model=model or None)
        if not llm_result:
            return None

        # Validate LLM results
        if field_type == "phone":
            cleaned = llm_result.replace(" ", "").replace("-", "")
            return cleaned if validate_phone_number(cleaned) else None
        elif field_type == "email":
            return llm_result.lower() if validate_email(llm_result.lower()) else None
        elif field_type == "preferred_date":
            parsed = parse_and_validate_date(llm_result)
            return parsed.isoformat() if parsed else None
        elif field_type == "name":
            return llm_result.title() if len(llm_result) >= 2 else None

    except Exception as exc:
        logger.warning("LLM extraction failed for %s: %s", field_type, exc)

    return None


def _extract_all_fields(
    message: str, existing_data: dict, model: str = ""
) -> dict[str, Optional[str]]:
    """
    Extract all possible appointment fields from a message.

    Uses regex first for each field. If any fields are still missing,
    falls back to LLM multi-field extraction for a single API call
    instead of one call per field.

    Never overwrites fields that already exist in existing_data.
    """
    # Start with regex multi-field scan
    regex_fields = _extract_fields_regex(message)

    # Merge: only fill in what's missing
    merged: dict[str, Optional[str]] = {}
    missing_after_regex = []

    for field in _REQUIRED_FIELDS:
        if existing_data.get(field):
            # Already have this field, keep it
            merged[field] = existing_data[field]
        elif regex_fields.get(field):
            # Regex found it
            merged[field] = regex_fields[field]
            logger.info("Regex extracted %s='%s'", field, regex_fields[field])
        else:
            merged[field] = None
            missing_after_regex.append(field)

    # If regex got everything, we're done
    if not missing_after_regex:
        return merged

    # LLM fallback: try multi-field extraction in one call
    try:
        from backend.services.llm_provider import extract_all_fields_with_llm
        llm_fields = extract_all_fields_with_llm(message, model=model or None)

        for field in missing_after_regex:
            llm_key = "date" if field == "preferred_date" else field
            llm_value = llm_fields.get(llm_key)

            if not llm_value:
                continue

            # Validate before accepting
            if field == "phone":
                cleaned = llm_value.replace(" ", "").replace("-", "")
                if validate_phone_number(cleaned):
                    merged[field] = cleaned
            elif field == "email":
                if validate_email(llm_value.lower()):
                    merged[field] = llm_value.lower()
            elif field == "preferred_date":
                parsed = parse_and_validate_date(llm_value)
                if parsed:
                    merged[field] = parsed.isoformat()
            elif field == "name":
                if len(llm_value) >= 2 and any(c.isalpha() for c in llm_value):
                    merged[field] = llm_value.title()

    except Exception as exc:
        logger.warning("LLM multi-field extraction failed: %s", exc)

    return merged


# ---------------------------------------------------------------------------
# Step management
# ---------------------------------------------------------------------------

def _find_next_missing_field(data: dict) -> Optional[str]:
    """Return the first required field that's missing from data, or None if all present."""
    for field in _REQUIRED_FIELDS:
        if not data.get(field):
            return field
    return None


def _prompt_for_field(field: str, data: dict) -> str:
    """Return the user-facing prompt for a specific missing field."""
    prompts = {
        "name": "What is your full name?",
        "phone": f"Thanks, {data.get('name', '')}! What is your phone number? (10 digits, starting with 98 or 97)",
        "email": "Got it. What is your email address?",
        "preferred_date": "When would you like to schedule the appointment? (e.g. 'next Monday', 'March 25th')",
    }
    return prompts.get(field, f"Please provide your {field}.")


def _format_confirmation(data: dict) -> str:
    """Format the confirmation message showing all collected data."""
    return (
        "Here's what I have:\n"
        f"- Name: {data.get('name', '')}\n"
        f"- Phone: {data.get('phone', '')}\n"
        f"- Email: {data.get('email', '')}\n"
        f"- Date: {data.get('preferred_date', '')}\n\n"
        "Shall I confirm this booking? (yes/no)"
    )


# ---------------------------------------------------------------------------
# Main handler
# ---------------------------------------------------------------------------

def handle_appointment(state: AgentState) -> AgentState:
    """
    LangGraph node: advance the appointment booking conversation.

    Key behaviors:
      - Extracts multiple fields from a single message (regex + LLM)
      - Skips steps for fields already collected
      - Only asks for what's still missing
      - Validates all fields before accepting
      - Never overwrites valid existing data
    """
    message = state.get("user_message", "").strip()
    step = state.get("appointment_step", "")
    data = dict(state.get("appointment_data", {}))
    llm_model = state.get("llm_model", "")

    # --- Start or restart the flow ----------------------------------------
    is_new_request = not step or step == "complete"
    is_stale_state = step and step != "complete" and not data
    if is_new_request or is_stale_state:
        data = {}
        logger.info("Appointment flow started (previous_step=%s)", step or "none")

        # Try regex first for initial field extraction
        regex_fields = _extract_fields_regex(message)
        for field, value in regex_fields.items():
            if value:
                data[field] = value

        # If regex found some but not all, try LLM for the rest
        missing_count = sum(1 for f in _REQUIRED_FIELDS if not data.get(f))
        if data and 0 < missing_count < len(_REQUIRED_FIELDS):
            try:
                extracted = _extract_all_fields(message, data, model=llm_model)
                for field, value in extracted.items():
                    if value and not data.get(field):
                        data[field] = value
            except Exception as exc:
                logger.warning("LLM multi-field extraction failed on start: %s", exc)

        next_missing = _find_next_missing_field(data)
        if next_missing is None:
            # All fields extracted from a single message — go to confirm
            state["appointment_step"] = "confirm"
            state["response"] = _format_confirmation(data)
        else:
            state["appointment_step"] = next_missing
            if data:
                # Some fields found — acknowledge and ask for next
                found_fields = [f for f in _REQUIRED_FIELDS if data.get(f)]
                ack = "Got it! I've captured: " + ", ".join(
                    f"**{f}**: {data[f]}" for f in found_fields
                ) + ". "
                state["response"] = ack + _prompt_for_field(next_missing, data)
            else:
                state["response"] = (
                    "Sure, I can help you book an appointment! "
                    + _prompt_for_field(next_missing, data)
                )

        state["appointment_data"] = data
        return state

    # --- Confirmation step ------------------------------------------------
    if step == "confirm":
        is_confirm = bool(_CONFIRM_PATTERN.search(message))
        is_deny = bool(_DENY_PATTERN.search(message))

        if is_confirm and not is_deny:
            try:
                appointment = AppointmentRequest(
                    name=data["name"],
                    phone=data["phone"],
                    email=data["email"],
                    preferred_date=data["preferred_date"],
                )
                result = book_appointment(appointment)
                state["response"] = (
                    f"{result.message}\n"
                    f"Your appointment ID is: {result.appointment_id}"
                )
                state["appointment_step"] = "complete"
                logger.info("Appointment booked: %s", result.appointment_id)

            except ValidationError as exc:
                error_messages = "; ".join(err["msg"] for err in exc.errors())
                state["response"] = (
                    f"There was a validation error: {error_messages}. "
                    "Let's start over — what is your full name?"
                )
                state["appointment_step"] = "name"
                state["appointment_data"] = {}
                logger.warning("Appointment validation failed: %s", error_messages)

        elif is_deny:
            state["response"] = (
                "No problem, the booking has been cancelled. "
                "Let me know if you need anything else!"
            )
            state["appointment_step"] = "complete"
            state["appointment_data"] = {}
            logger.info("User cancelled appointment booking")
        else:
            state["response"] = "Please reply with 'yes' to confirm or 'no' to cancel."

        state["appointment_data"] = data
        return state

    # --- Collecting a specific field --------------------------------------
    # Step 1: Try regex-only multi-field scan (no LLM, fast)
    regex_fields = _extract_fields_regex(message)
    for field in _REQUIRED_FIELDS:
        if not data.get(field) and regex_fields.get(field):
            data[field] = regex_fields[field]
            logger.info("Regex extracted %s='%s'", field, regex_fields[field])

    # Step 2: If we still don't have the expected field, try focused single-field
    # extraction (regex + LLM fallback)
    if not data.get(step):
        single_result = _extract_single_field(message, step, model=llm_model)
        if single_result:
            data[step] = single_result

    # If we still don't have the expected field, prompt again with an error
    if not data.get(step):
        error_messages = {
            "name": "That doesn't look like a valid name. Please enter your full name.",
            "phone": "That phone number doesn't look right. Please enter a valid Nepali phone number (10 digits, starting with 98 or 97).",
            "email": "That doesn't look like a valid email address. Please try again.",
            "preferred_date": (
                "I couldn't understand that date, or it's in the past. "
                "Please try again (e.g. 'next Friday', 'April 10')."
            ),
        }
        state["response"] = error_messages.get(step, f"Please provide a valid {step}.")
        state["appointment_data"] = data
        return state

    # Find the next missing field
    next_missing = _find_next_missing_field(data)

    if next_missing is None:
        # All fields collected — show confirmation
        state["appointment_step"] = "confirm"
        state["response"] = _format_confirmation(data)
    else:
        state["appointment_step"] = next_missing
        state["response"] = _prompt_for_field(next_missing, data)

    state["appointment_data"] = data
    return state
