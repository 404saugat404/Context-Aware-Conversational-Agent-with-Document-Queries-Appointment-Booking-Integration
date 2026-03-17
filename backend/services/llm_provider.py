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

from backend.config import GEMINI_API_KEY, GEMINI_MODEL_NAME, OLLAMA_BASE_URL
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


def _call_gemini(prompt: str, system_instruction: str, model: str) -> str:
    """Call Gemini API and return response text."""
    logger.info("Calling Gemini model: %s", model)
    client = _get_gemini_client()
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=genai.types.GenerateContentConfig(
            system_instruction=system_instruction,
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


def _call_ollama(prompt: str, system_instruction: str, model: str) -> str:
    """Call Ollama's /api/generate endpoint and return the response text."""
    logger.info("Calling Ollama model: %s at %s", model, OLLAMA_BASE_URL)

    payload = {
        "model": model,
        "system": system_instruction,
        "prompt": prompt,
        "stream": False,
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

def generate_response(
    prompt: str,
    system_instruction: str,
    model: Optional[str] = None,
    provider: Optional[str] = None,
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

    Returns:
        The LLM's response text.
    """
    resolved_model = model or GEMINI_MODEL_NAME

    if provider:
        resolved_provider = LLMProvider(provider)
    else:
        resolved_provider = detect_provider(resolved_model)

    logger.info("LLM request — provider=%s, model=%s", resolved_provider.value, resolved_model)

    if resolved_provider == LLMProvider.GEMINI:
        return _call_gemini(prompt, system_instruction, resolved_model)
    else:
        return _call_ollama(prompt, system_instruction, resolved_model)
