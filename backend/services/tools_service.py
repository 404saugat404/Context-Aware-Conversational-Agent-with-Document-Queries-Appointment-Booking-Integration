"""
Utility helpers: date parsing, phone validation, query rewriting.

These are reusable building blocks used by the agents and services.
"""

from __future__ import annotations

import re
from datetime import date, datetime
from typing import Optional

import dateparser
import parsedatetime

from backend.logger import get_logger

logger = get_logger(__name__)

# Shared parsedatetime calendar instance
_pdt_calendar = parsedatetime.Calendar()


def _parse_with_dateparser(text: str) -> Optional[date]:
    """Attempt parsing with dateparser (handles absolute dates well)."""
    parsed = dateparser.parse(
        text,
        settings={
            "PREFER_DATES_FROM": "future",
            "STRICT_PARSING": False,
        },
    )
    if parsed is not None:
        return parsed.date()
    return None


def _parse_with_parsedatetime(text: str) -> Optional[date]:
    """Attempt parsing with parsedatetime (handles relative dates like 'next Monday')."""
    time_struct, status = _pdt_calendar.parse(text)
    if status == 0:
        return None
    return datetime(*time_struct[:6]).date()


def parse_natural_language_date(text: str) -> Optional[date]:
    """
    Parse a human-friendly date string into a ``datetime.date``.

    Uses dateparser first, then falls back to parsedatetime for relative
    expressions like "next Monday" or "coming Tuesday" that dateparser
    may not handle.

    Handles inputs like "next Monday", "March 5th", "tomorrow", "2025-04-01".

    Returns:
        The parsed date, or ``None`` if parsing fails.
    """
    if not text or not text.strip():
        logger.debug("parse_natural_language_date received empty input")
        return None

    # Try dateparser first (good for absolute dates and many relative ones)
    result = _parse_with_dateparser(text)
    if result is not None:
        logger.debug("Parsed '%s' -> %s (via dateparser)", text, result.isoformat())
        return result

    # Fall back to parsedatetime (handles "next Monday", "coming Tuesday", etc.)
    result = _parse_with_parsedatetime(text)
    if result is not None:
        logger.debug("Parsed '%s' -> %s (via parsedatetime)", text, result.isoformat())
        return result

    logger.warning("Could not parse date from text: '%s'", text)
    return None


def validate_phone_number(phone: str) -> bool:
    """Return True if *phone* is a valid Nepali mobile number (starts with 98 or 97, 10 digits, optional +977)."""
    pattern = r"^(?:\+977)?(98|97)\d{8}$"
    is_valid = bool(re.match(pattern, phone.strip()))
    if not is_valid:
        logger.debug("Invalid phone number: '%s'", phone)
    return is_valid


def validate_email(email: str) -> bool:
    """Basic email format validation (not exhaustive — Pydantic EmailStr is authoritative)."""
    pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    is_valid = bool(re.match(pattern, email.strip()))
    if not is_valid:
        logger.debug("Invalid email format: '%s'", email)
    return is_valid


def rewrite_query_for_retrieval(query: str) -> str:
    """
    Lightly rewrite the user query to improve retrieval quality.

    Current heuristics:
      - Strip filler phrases ("can you tell me", "I'd like to know").
      - Collapse whitespace.

    This is intentionally simple; a more advanced version would use an LLM.
    """
    filler_phrases = [
        r"(?i)^(can you |could you |please |tell me |I('d| would) like to know )",
        r"(?i)^(what is |what are |how do I |how can I )",
    ]
    cleaned = query.strip()
    for pattern in filler_phrases:
        cleaned = re.sub(pattern, "", cleaned).strip()

    # Collapse multiple spaces
    cleaned = re.sub(r"\s+", " ", cleaned)

    if cleaned != query.strip():
        logger.debug("Rewrote query: '%s' -> '%s'", query.strip(), cleaned)

    return cleaned if cleaned else query.strip()
