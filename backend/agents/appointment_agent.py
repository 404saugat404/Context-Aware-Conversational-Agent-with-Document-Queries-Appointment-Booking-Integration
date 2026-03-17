"""
Appointment booking agent node for LangGraph.

Drives a multi-step conversational form collecting: name, phone, email, date.
Each invocation advances one step and prompts for the next field.
"""

from __future__ import annotations

from pydantic import ValidationError

from backend.agents.state import AgentState
from backend.logger import get_logger
from backend.models.appointment_schema import AppointmentRequest
from backend.services.appointment_service import book_appointment, parse_and_validate_date
from backend.services.tools_service import validate_email, validate_phone_number

logger = get_logger(__name__)

# The order in which we collect fields
_BOOKING_STEPS = ["name", "phone", "email", "date", "confirm"]

_STEP_PROMPTS = {
    "name": "Sure, I can help you book an appointment! What is your full name?",
    "phone": "Thanks, {name}! What is your phone number? (10-15 digits)",
    "email": "Got it. What is your email address?",
    "date": "When would you like to schedule the appointment? (e.g. 'next Monday', 'March 25th')",
    "confirm": (
        "Here's what I have:\n"
        "- Name: {name}\n"
        "- Phone: {phone}\n"
        "- Email: {email}\n"
        "- Date: {preferred_date}\n\n"
        "Shall I confirm this booking? (yes/no)"
    ),
}


def _next_step(current: str) -> str:
    """Return the step that follows *current*, or 'complete' if done."""
    try:
        idx = _BOOKING_STEPS.index(current)
        return _BOOKING_STEPS[idx + 1] if idx + 1 < len(_BOOKING_STEPS) else "complete"
    except ValueError:
        return _BOOKING_STEPS[0]


def handle_appointment(state: AgentState) -> AgentState:
    """
    LangGraph node: advance the appointment booking conversation by one step.

    On each call the agent inspects ``appointment_step``, validates the user's
    input for that step, stores it in ``appointment_data``, and moves to the
    next step.
    """
    message = state.get("user_message", "").strip()
    step = state.get("appointment_step", "")
    data = state.get("appointment_data", {})

    # First contact — start the flow
    if not step:
        state["appointment_step"] = "name"
        state["appointment_data"] = {}
        state["response"] = _STEP_PROMPTS["name"]
        logger.info("Appointment flow started")
        return state

    # --- Collect & validate each field ---------------------------------

    if step == "name":
        if len(message) < 2 or not any(c.isalpha() for c in message):
            state["response"] = "That doesn't look like a valid name. Please enter your full name."
            return state
        data["name"] = message.strip()
        state["appointment_step"] = "phone"
        state["response"] = _STEP_PROMPTS["phone"].format(**data)

    elif step == "phone":
        cleaned_phone = message.replace(" ", "").replace("-", "")
        if not validate_phone_number(cleaned_phone):
            state["response"] = (
                "That phone number doesn't look right. "
                "Please enter a valid phone number (10-15 digits)."
            )
            return state
        data["phone"] = cleaned_phone
        state["appointment_step"] = "email"
        state["response"] = _STEP_PROMPTS["email"]

    elif step == "email":
        if not validate_email(message):
            state["response"] = "That doesn't look like a valid email address. Please try again."
            return state
        data["email"] = message.strip()
        state["appointment_step"] = "date"
        state["response"] = _STEP_PROMPTS["date"]

    elif step == "date":
        parsed_date = parse_and_validate_date(message)
        if parsed_date is None:
            state["response"] = (
                "I couldn't understand that date, or it's in the past. "
                "Please try again (e.g. 'next Friday', 'April 10')."
            )
            return state
        data["preferred_date"] = parsed_date.isoformat()
        state["appointment_step"] = "confirm"
        state["response"] = _STEP_PROMPTS["confirm"].format(**data)

    elif step == "confirm":
        if message.lower() in ("yes", "y", "confirm", "sure", "ok"):
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
                error_messages = "; ".join(
                    err["msg"] for err in exc.errors()
                )
                state["response"] = (
                    f"There was a validation error: {error_messages}. "
                    "Let's start over — what is your full name?"
                )
                state["appointment_step"] = "name"
                state["appointment_data"] = {}
                logger.warning("Appointment validation failed: %s", error_messages)

        elif message.lower() in ("no", "n", "cancel", "nope"):
            state["response"] = "No problem, the booking has been cancelled. Let me know if you need anything else!"
            state["appointment_step"] = "complete"
            state["appointment_data"] = {}
            logger.info("User cancelled appointment booking")
        else:
            state["response"] = "Please reply with 'yes' to confirm or 'no' to cancel."

    state["appointment_data"] = data
    return state
