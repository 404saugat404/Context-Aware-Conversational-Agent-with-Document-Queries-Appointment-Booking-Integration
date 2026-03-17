"""
Pydantic schemas for appointment booking.

Validates user-supplied appointment data with strict rules
for phone numbers, email, and date formats.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Optional

from pydantic import BaseModel, EmailStr, Field, field_validator


class AppointmentRequest(BaseModel):
    """Validated appointment booking request from the user."""

    name: str = Field(
        ...,
        min_length=2,
        max_length=100,
        description="Full name of the person booking the appointment",
    )
    phone: str = Field(
        ...,
        pattern=r"^\+?\d{10,15}$",
        description="Phone number (10-15 digits, optional leading +)",
    )
    email: EmailStr = Field(
        ...,
        description="Valid email address",
    )
    preferred_date: date = Field(
        ...,
        description="Preferred appointment date (must be today or later)",
    )
    reason: Optional[str] = Field(
        default=None,
        max_length=500,
        description="Optional reason or notes for the appointment",
    )

    @field_validator("preferred_date")
    @classmethod
    def date_must_not_be_in_past(cls, value: date) -> date:
        if value < date.today():
            raise ValueError("Preferred date cannot be in the past")
        return value

    @field_validator("name")
    @classmethod
    def name_must_contain_letters(cls, value: str) -> str:
        if not any(c.isalpha() for c in value):
            raise ValueError("Name must contain at least one letter")
        return value.strip()


class AppointmentResponse(BaseModel):
    """Response returned after an appointment is processed."""

    success: bool
    message: str
    appointment_id: Optional[str] = None
    appointment_details: Optional[AppointmentRequest] = None


class AppointmentSlot(BaseModel):
    """Represents a single available appointment slot."""

    date: date
    time_slot: str = Field(..., description="e.g. '09:00-09:30'")
    is_available: bool = True


class ChatRequest(BaseModel):
    """Incoming chat message from the user."""

    message: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The user's message",
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Session identifier for conversation continuity",
    )


class ChatResponse(BaseModel):
    """Outgoing chat response to the user."""

    reply: str
    session_id: str
    intent: Optional[str] = Field(
        default=None,
        description="Detected intent (rag, appointment, greeting, etc.)",
    )
    sources: Optional[list[str]] = Field(
        default=None,
        description="Source documents used for RAG responses",
    )
