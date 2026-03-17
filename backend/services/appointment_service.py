"""
Appointment booking service.

Handles validation, date parsing, and persistence of appointment records.
Currently uses in-memory storage; swap in a database adapter for production.
"""

from __future__ import annotations

import uuid
from datetime import date
from typing import Dict, List, Optional

from backend.logger import get_logger
from backend.models.appointment_schema import (
    AppointmentRequest,
    AppointmentResponse,
    AppointmentSlot,
)
from backend.services.tools_service import parse_natural_language_date

logger = get_logger(__name__)

# In-memory appointment store (replace with database in production)
_appointment_store: Dict[str, AppointmentRequest] = {}


def book_appointment(appointment_data: AppointmentRequest) -> AppointmentResponse:
    """
    Validate and persist an appointment.

    Pydantic has already validated the schema by this point, so this
    function handles business-level checks (e.g. slot availability).

    Args:
        appointment_data: Validated appointment request.

    Returns:
        An ``AppointmentResponse`` indicating success or failure.
    """
    logger.info(
        "Booking appointment for %s on %s",
        appointment_data.name,
        appointment_data.preferred_date.isoformat(),
    )

    appointment_id = uuid.uuid4().hex[:12]
    _appointment_store[appointment_id] = appointment_data

    logger.info("Appointment booked successfully: id=%s", appointment_id)
    return AppointmentResponse(
        success=True,
        message=f"Appointment booked successfully for {appointment_data.name} "
        f"on {appointment_data.preferred_date.isoformat()}.",
        appointment_id=appointment_id,
        appointment_details=appointment_data,
    )


def parse_and_validate_date(date_text: str) -> Optional[date]:
    """
    Parse a natural language date and validate it is not in the past.

    Args:
        date_text: Free-text date input (e.g. "next Tuesday").

    Returns:
        A valid ``date`` object, or ``None`` if parsing/validation fails.
    """
    parsed = parse_natural_language_date(date_text)
    if parsed is None:
        logger.warning("Failed to parse date text: '%s'", date_text)
        return None

    if parsed < date.today():
        logger.warning("Parsed date %s is in the past", parsed.isoformat())
        return None

    return parsed


def get_available_slots(target_date: date) -> List[AppointmentSlot]:
    """
    Return available appointment slots for a given date.

    This is a stub — replace with real calendar/scheduling logic.
    """
    if target_date < date.today():
        logger.debug("Requested slots for past date %s", target_date.isoformat())
        return []

    # Stub: generate 30-minute slots from 09:00 to 17:00
    slots: list[AppointmentSlot] = []
    for hour in range(9, 17):
        for minute in (0, 30):
            time_slot = f"{hour:02d}:{minute:02d}-{hour:02d}:{minute + 30:02d}"
            slots.append(
                AppointmentSlot(
                    date=target_date,
                    time_slot=time_slot,
                    is_available=True,
                )
            )

    logger.debug("Generated %d slots for %s", len(slots), target_date.isoformat())
    return slots


def get_appointment_by_id(appointment_id: str) -> Optional[AppointmentRequest]:
    """Look up a previously booked appointment by its ID."""
    return _appointment_store.get(appointment_id)
