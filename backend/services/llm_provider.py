"""
Unified LLM provider service.

Abstracts over Gemini and Ollama so the rest of the codebase can call
``generate_response()`` without caring which backend is active.

Usage:
    from backend.services.llm_provider import generate_response

    answer = generate_response(
        prompt="What services do you offer?",
        system_instruction="Answer from context only...",
        model="mistral",           # or "gemini-2.0-flash"
        provider="ollama",         # or "gemini"
    )
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

import httpx
from google import genai

from backend.config import GEMINI_API_KEY, GEMINI_MODEL_NAME, LLM_TEMPERATURE, OLLAMA_BASE_URL
from backend.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Provider enum
# ---------------------------------------------------------------------------

class LLMProvider(str, Enum):
    """Supported LLM providers."""

    GEMINI = "gemini"
    OLLAMA = "ollama"


# ---------------------------------------------------------------------------
# Model registry — maps model names to their provider
# ---------------------------------------------------------------------------

# Gemini models (cloud)
_GEMINI_MODELS = {
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
}

# Well-known Ollama models (local) — this isn't exhaustive, any model
# pulled into Ollama will work. This list is used by the /models endpoint.
_OLLAMA_KNOWN_MODELS = {
    "mistral",
    "llama3.2:1b",
    "llama3.2:3b",
    "llama3.1:8b",
    "gemma2:2b",
    "gemma2:9b",
    "phi3:mini",
    "qwen2.5:7b",
}


def detect_provider(model: str) -> LLMProvider:
    """Infer the provider from the model name."""
    if model.startswith("gemini"):
        return LLMProvider.GEMINI
    return LLMProvider.OLLAMA


def list_available_models() -> dict:
    """
    Return actually available models grouped by provider.

    Ollama models are only listed if Ollama is running and has models pulled.
    """
    ollama_local = _get_ollama_local_models()
    ollama_status = "connected" if ollama_local else "not_running"

    if not ollama_local:
        logger.info("Ollama is not reachable or has no models pulled")

    return {
        "gemini": sorted(_GEMINI_MODELS),
        "ollama": sorted(ollama_local),
        "ollama_status": ollama_status,
        "default_model": GEMINI_MODEL_NAME,
    }


# ---------------------------------------------------------------------------
# Gemini client (initialised lazily)
# ---------------------------------------------------------------------------

_gemini_client: Optional[genai.Client] = None


def _get_gemini_client() -> genai.Client:
    global _gemini_client
    if _gemini_client is None:
        if not GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY is not set")
        _gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    return _gemini_client


def _call_gemini(prompt: str, system_instruction: str, model: str, temperature: float = LLM_TEMPERATURE) -> str:
    """Call Gemini API and return response text."""
    logger.info("Calling Gemini model: %s (temperature=%.2f)", model, temperature)
    client = _get_gemini_client()
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=genai.types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=temperature,
        ),
    )
    return response.text


# ---------------------------------------------------------------------------
# Ollama client (uses HTTP API directly — no extra dependency needed)
# ---------------------------------------------------------------------------

def _get_ollama_local_models() -> list[str]:
    """Query Ollama for locally available models. Returns empty list if unreachable."""
    try:
        resp = httpx.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3.0)
        resp.raise_for_status()
        models = resp.json().get("models", [])
        return [m["name"] for m in models]
    except Exception:
        return []


def _call_ollama(prompt: str, system_instruction: str, model: str, temperature: float = LLM_TEMPERATURE) -> str:
    """Call Ollama's /api/generate endpoint and return the response text."""
    logger.info("Calling Ollama model: %s at %s (temperature=%.2f)", model, OLLAMA_BASE_URL, temperature)

    payload = {
        "model": model,
        "system": system_instruction,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
        },
    }

    try:
        resp = httpx.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json=payload,
            timeout=120.0,  # local models can be slow on CPU
        )
        resp.raise_for_status()
        result = resp.json()
        return result.get("response", "")
    except httpx.ConnectError:
        logger.error("Cannot connect to Ollama at %s. Is it running?", OLLAMA_BASE_URL)
        raise RuntimeError(
            f"Cannot connect to Ollama at {OLLAMA_BASE_URL}. "
            "Make sure Ollama is running (run: ollama serve)."
        )
    except httpx.HTTPStatusError as exc:
        logger.error("Ollama returned error: %s", exc.response.text)
        raise RuntimeError(f"Ollama error: {exc.response.text}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# LLM-based field extraction prompts
# ---------------------------------------------------------------------------

_EXTRACT_FIELD_PROMPTS = {
    "name": (
        'Extract ONLY the person\'s name from this message. '
        'Return just the name, nothing else. If no name is found, return "NONE".\n\n'
        'Message: "{message}"\n\nName:'
    ),
    "phone": (
        'Extract ONLY the phone number from this message. '
        'Return just the digits (with optional leading +), nothing else. '
        'If no phone number is found, return "NONE".\n\n'
        'Message: "{message}"\n\nPhone:'
    ),
    "email": (
        'Extract ONLY the email address from this message. '
        'Return just the email, nothing else. If no email is found, return "NONE".\n\n'
        'Message: "{message}"\n\nEmail:'
    ),
}

_EXTRACT_ALL_FIELDS_PROMPT = """Extract appointment details from this user message.
Return ONLY a JSON object with the fields you can find. Use null for fields not present.

Fields to extract:
- name: The person's full name (string or null)
- phone: Phone number as digits only, with optional leading + (string or null)
- email: Email address (string or null)
- date: The preferred date in natural language exactly as stated (string or null)

Rules:
- Return ONLY valid JSON, no explanation or markdown
- Only extract fields that are clearly present in the message
- Do not guess or infer missing fields
- For date, preserve the user's wording (e.g. "next Monday", "March 25th")

Message: "{message}"

JSON:"""


def extract_field_with_llm(
    message: str, field_type: str, model: Optional[str] = None
) -> Optional[str]:
    """
    Use the LLM to extract a specific field value from natural language input.

    Args:
        message: The raw user message.
        field_type: One of "name", "phone", "email".
        model: LLM model to use (auto-detected if None).

    Returns:
        The extracted value, or None if extraction failed.
    """
    prompt_template = _EXTRACT_FIELD_PROMPTS.get(field_type)
    if not prompt_template:
        logger.warning("No extraction prompt for field type: %s", field_type)
        return None

    resolved_model = model or GEMINI_MODEL_NAME
    prompt = prompt_template.format(message=message)
    system_instruction = (
        "You are a data extraction assistant. "
        "Return ONLY the requested value, no explanation or extra text."
    )

    try:
        raw = generate_response(
            prompt=prompt,
            system_instruction=system_instruction,
            model=resolved_model,
        ).strip()

        cleaned = raw.strip("\"'`.").strip()

        if not cleaned or cleaned.upper() == "NONE":
            logger.info("LLM could not extract %s from: %.60s", field_type, message)
            return None

        logger.info("LLM extracted %s='%s' from: %.60s", field_type, cleaned, message)
        return cleaned
    except Exception as exc:
        logger.error("LLM field extraction failed for %s: %s", field_type, exc)
        return None


def extract_all_fields_with_llm(
    message: str, model: Optional[str] = None
) -> dict[str, Optional[str]]:
    """
    Extract all appointment fields from a single message using the LLM.

    Returns a dict with keys: name, phone, email, date.
    Values are the extracted strings or None if not found.
    """
    import json

    resolved_model = model or GEMINI_MODEL_NAME
    prompt = _EXTRACT_ALL_FIELDS_PROMPT.format(message=message)
    system_instruction = (
        "You are a data extraction assistant. "
        "Return ONLY valid JSON, no explanation or extra text."
    )

    empty_result: dict[str, Optional[str]] = {
        "name": None, "phone": None, "email": None, "date": None,
    }

    try:
        raw = generate_response(
            prompt=prompt,
            system_instruction=system_instruction,
            model=resolved_model,
        ).strip()

        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
            if raw.endswith("```"):
                raw = raw[:-3]
            raw = raw.strip()

        parsed = json.loads(raw)

        result: dict[str, Optional[str]] = {}
        for key in ("name", "phone", "email", "date"):
            value = parsed.get(key)
            if value and str(value).strip().upper() != "NONE":
                result[key] = str(value).strip()
            else:
                result[key] = None

        logger.info(
            "LLM multi-field extraction: %s from: %.80s",
            {k: v for k, v in result.items() if v},
            message,
        )
        return result

    except json.JSONDecodeError as exc:
        logger.warning("LLM returned invalid JSON for multi-field extraction: %s", exc)
        return empty_result
    except Exception as exc:
        logger.error("LLM multi-field extraction failed: %s", exc)
        return empty_result


_INTENT_CLASSIFICATION_PROMPT = """Classify the following user message into exactly ONE of these categories:
- rag: The user is asking a question, seeking information, or wants to know something.
- appointment: The user wants to book, schedule, or set up an appointment/meeting/consultation.
- greeting: The user is greeting (hi, hello, hey, good morning, etc.) or making small talk.
- appointment_cancel: The user wants to cancel or stop an ongoing appointment booking.

Rules:
- Return ONLY the category label (rag, appointment, greeting, or appointment_cancel).
- Do NOT add any explanation, punctuation, or extra text.
- If unsure, return "rag".

User message: "{message}"

Category:"""


def classify_intent_with_llm(message: str, model: Optional[str] = None) -> Optional[str]:
    """
    Use the LLM to classify intent when regex is uncertain.

    Returns one of: rag, appointment, greeting, appointment_cancel, or None on failure.
    """
    resolved_model = model or GEMINI_MODEL_NAME
    prompt = _INTENT_CLASSIFICATION_PROMPT.format(message=message)
    system_instruction = "You are an intent classifier. Return only a single label, nothing else."

    try:
        raw = generate_response(
            prompt=prompt,
            system_instruction=system_instruction,
            model=resolved_model,
        ).strip().lower()

        # Extract just the label — LLMs sometimes add quotes or extra whitespace
        valid_intents = {"rag", "appointment", "greeting", "appointment_cancel"}
        for intent in valid_intents:
            if intent in raw:
                logger.info("LLM classified intent as '%s' (raw: '%s')", intent, raw)
                return intent

        logger.warning("LLM returned unrecognized intent: '%s', defaulting to None", raw)
        return None
    except Exception as exc:
        logger.error("LLM intent classification failed: %s", exc)
        return None


def generate_response(
    prompt: str,
    system_instruction: str,
    model: Optional[str] = None,
    provider: Optional[str] = None,
    temperature: float = LLM_TEMPERATURE,
) -> str:
    """
    Generate a response from the configured LLM.

    Args:
        prompt: The user prompt / question.
        system_instruction: System-level instructions for the LLM.
        model: Model name (e.g. "mistral", "gemini-2.0-flash").
               Defaults to GEMINI_MODEL_NAME from config.
        provider: Force a specific provider ("gemini" or "ollama").
                  If None, auto-detected from the model name.
        temperature: Sampling temperature (0.0 = deterministic, 1.0+ = creative).
                     Defaults to LLM_TEMPERATURE from config.

    Returns:
        The LLM's response text.
    """
    resolved_model = model or GEMINI_MODEL_NAME

    if provider:
        resolved_provider = LLMProvider(provider)
    else:
        resolved_provider = detect_provider(resolved_model)

    logger.info("LLM request — provider=%s, model=%s, temperature=%.2f", resolved_provider.value, resolved_model, temperature)

    if resolved_provider == LLMProvider.GEMINI:
        return _call_gemini(prompt, system_instruction, resolved_model, temperature)
    else:
        return _call_ollama(prompt, system_instruction, resolved_model, temperature)
