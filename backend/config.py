"""
Centralized configuration loaded from environment variables.

All settings are read once at import time via python-dotenv.
Modules should import constants from here rather than calling os.getenv directly.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Qdrant
# ---------------------------------------------------------------------------
QDRANT_URL: str = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION_NAME: str = os.getenv("QDRANT_COLLECTION_NAME", "documents")

# ---------------------------------------------------------------------------
# LLM providers
# ---------------------------------------------------------------------------
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL_NAME: str = os.getenv("GEMINI_MODEL_NAME", "gemini-2.0-flash")
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_API_KEY: str = os.getenv("OLLAMA_API_KEY", "")
LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.7"))

# ---------------------------------------------------------------------------
# Embedding & reranker model identifiers
# ---------------------------------------------------------------------------
EMBEDDING_MODEL_NAME: str = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-small-en-v1.5")
RERANKER_MODEL_NAME: str = os.getenv("RERANKER_MODEL_NAME", "BAAI/bge-reranker-base")

# ---------------------------------------------------------------------------
# RAG retrieval parameters
# ---------------------------------------------------------------------------
RAG_TOP_K: int = int(os.getenv("RAG_TOP_K", "10"))
RAG_RERANK_TOP_N: int = int(os.getenv("RAG_RERANK_TOP_N", "3"))
RAG_MAX_CONTEXT_TOKENS: int = int(os.getenv("RAG_MAX_CONTEXT_TOKENS", "2048"))

# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
DATA_DIR: str = os.getenv("DATA_DIR", os.path.join(os.path.dirname(os.path.dirname(__file__)), "data"))

# ---------------------------------------------------------------------------
# Email
# ---------------------------------------------------------------------------
MAIL_USERNAME: str = os.getenv("MAIL_USERNAME", "")
MAIL_PASSWORD: str = os.getenv("MAIL_PASSWORD", "")
MAIL_FROM: str = os.getenv("MAIL_FROM", "")
MAIL_PORT: int = int(os.getenv("MAIL_PORT", "587"))
MAIL_SERVER: str = os.getenv("MAIL_SERVER", "smtp.gmail.com")
MAIL_STARTTLS: bool = os.getenv("MAIL_STARTTLS", "true").lower() == "true"
MAIL_SSL_TLS: bool = os.getenv("MAIL_SSL_TLS", "false").lower() == "true"
USE_CREDENTIALS: bool = os.getenv("USE_CREDENTIALS", "true").lower() == "true"
TEMPLATE_FOLDER: str = os.getenv(
    "TEMPLATE_FOLDER",
    os.path.join(os.path.dirname(__file__), "templates"),
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
