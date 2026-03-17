"""
JSON file-based persistence for conversations and appointments.

Stores data in the DATA_DIR directory as JSON files:
  - conversations.json: session_id -> chat history
  - appointments.json:  appointment_id -> appointment details

Thread-safe via file-level locking (fcntl). Reads load the full file;
writes flush atomically (write-to-temp, then rename).
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import date, datetime
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional

from backend.config import DATA_DIR
from backend.logger import get_logger

logger = get_logger(__name__)

# Ensure the data directory exists at import time
Path(DATA_DIR).mkdir(parents=True, exist_ok=True)

_CONVERSATIONS_FILE = os.path.join(DATA_DIR, "conversations.json")
_APPOINTMENTS_FILE = os.path.join(DATA_DIR, "appointments.json")

# In-process locks to prevent concurrent writes from async workers
_conversations_lock = Lock()
_appointments_lock = Lock()


# ---------------------------------------------------------------------------
# JSON serialization helpers
# ---------------------------------------------------------------------------

class _DateAwareEncoder(json.JSONEncoder):
    """Serialize date/datetime objects to ISO format strings."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, (date, datetime)):
            return obj.isoformat()
        return super().default(obj)


def _read_json_file(filepath: str) -> dict:
    """Read and parse a JSON file. Returns empty dict if file doesn't exist or is corrupt."""
    if not os.path.exists(filepath):
        return {}
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError) as exc:
        logger.error("Failed to read %s: %s", filepath, exc)
        return {}


def _write_json_file(filepath: str, data: dict) -> None:
    """
    Write data to a JSON file atomically.

    Writes to a temporary file first, then renames to avoid partial writes
    if the process is interrupted.
    """
    dir_name = os.path.dirname(filepath)
    try:
        fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, cls=_DateAwareEncoder, ensure_ascii=False)
        os.replace(tmp_path, filepath)
    except OSError as exc:
        logger.error("Failed to write %s: %s", filepath, exc)
        # Clean up temp file on failure
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Conversation persistence
# ---------------------------------------------------------------------------

def load_all_conversations() -> Dict[str, List[Dict[str, str]]]:
    """Load all conversations from disk. Returns {session_id: [messages]}."""
    with _conversations_lock:
        data = _read_json_file(_CONVERSATIONS_FILE)
    logger.info("Loaded %d conversations from disk", len(data))
    return data


def save_conversation(session_id: str, chat_history: List[Dict[str, str]]) -> None:
    """Persist a single session's chat history to disk."""
    with _conversations_lock:
        all_conversations = _read_json_file(_CONVERSATIONS_FILE)
        all_conversations[session_id] = chat_history
        _write_json_file(_CONVERSATIONS_FILE, all_conversations)
    logger.debug("Saved conversation for session %s (%d messages)", session_id, len(chat_history))


def load_conversation(session_id: str) -> Optional[List[Dict[str, str]]]:
    """Load a single session's chat history. Returns None if not found."""
    with _conversations_lock:
        all_conversations = _read_json_file(_CONVERSATIONS_FILE)
    return all_conversations.get(session_id)


# ---------------------------------------------------------------------------
# Appointment persistence
# ---------------------------------------------------------------------------

def load_all_appointments() -> Dict[str, dict]:
    """Load all appointments from disk. Returns {appointment_id: details}."""
    with _appointments_lock:
        data = _read_json_file(_APPOINTMENTS_FILE)
    logger.info("Loaded %d appointments from disk", len(data))
    return data


def save_appointment(appointment_id: str, appointment_data: dict) -> None:
    """Persist a single appointment to disk."""
    with _appointments_lock:
        all_appointments = _read_json_file(_APPOINTMENTS_FILE)
        all_appointments[appointment_id] = appointment_data
        _write_json_file(_APPOINTMENTS_FILE, all_appointments)
    logger.info("Saved appointment %s to disk", appointment_id)


def load_appointment(appointment_id: str) -> Optional[dict]:
    """Load a single appointment by ID. Returns None if not found."""
    with _appointments_lock:
        all_appointments = _read_json_file(_APPOINTMENTS_FILE)
    return all_appointments.get(appointment_id)
