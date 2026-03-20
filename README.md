# Context-Aware Conversational Agent

A production-ready conversational chatbot that seamlessly handles **document-based Q&A (RAG)** and **appointment booking** through a unified chat interface. Built with FastAPI, LangGraph, Qdrant, and a **dual LLM backend** (Google Gemini + Ollama for local models).

---

## System Architecture

```
                                    +---------------------+
                                    |    FastAPI Server    |
                                    |   (backend/main.py) |
                                    +----------+----------+
                                               |
                              +----------------+----------------+
                              |                |                |
                        POST /chat      POST /appointment  POST /ingest
                              |                |                |
                              v                v                v
                    +---------+------+   +-----+------+   +----+-------+
                    | Orchestrator   |   | Direct     |   | Ingestion  |
                    | (LangGraph)    |   | Booking    |   | Pipeline   |
                    +--------+-------+   +-----+------+   +----+-------+
                             |                 |                |
                             v                 |                v
              +-----------------------------+  |    +-----------------------+
              |      Intent Classifier      |  |    | chunk -> embed ->     |
              | (regex + LLM + context)    |  |    | upsert into Qdrant    |
              +----+------+------+----+----+  |    +-----------------------+
                   |      |      |    |       |
          +--------+  +---+  +---+  +-+-------+--+
          |           |      |      |             |
          v           v      v      v             |
    +-----+---+ +----+--+ +-+----+ +----------+  |
    |   RAG   | | Appt  | |Greet | | Cancel   |  |
    |  Agent  | | Agent | |Agent | | Handler  |  |
    +----+----+ +---+---+ +--+---+ +-----+----+  |
         |          |         |          |        |
         v          |         |          |        |
    +----+----+     |         |          |        |
    | RAG     |     v         |          |        |
    | Service |  +--+------+  |          |        |
    +----+----+  | Tools   |  |          |        |
         |       | Service |  |          |        |
    +----+----+  +----+----+  |          |        |
    |Retriever|       |       |          |        |
    +----+----+  +----+----+  |          |        |
         |       |dateparser|  |          |        |
    +----+----+  +---------+  |          |        |
    |Reranker |               |          |        |
    +----+----+               |          |        |
         |                    |          |        |
    +----+--------+           |          |        |
    |Context      |           |          |        |
    |Builder      |           |          |        |
    +----+--------+           |          |        |
         |                    |          |        |
         v                    |          |        |
    +----+----+               |          |        |
    | LLM     |               |          |        |
    | Gemini/ |               |          |        |
    | Ollama  |               |          |        |
    +---------+               |          |        |
                              v          v        v
                         +----+----------+--------+----+
                         |     In-Memory Session       |
                         |  (chat history, appt state) |
                         +-----------------------------+
```

---

## LangGraph Agent Flow

```
                    +-------------------+
                    |   Entry Point     |
                    | (user message in) |
                    +---------+---------+
                              |
                              v
                    +---------+---------+
                    | classify_intent   |
                    | (regex + context  |
                    |  + LLM fallback)  |
                    +-+----+----+----+--+
                      |    |    |    |
            rag       |    |    |    |  appointment_cancel
           +----------+    |    |    +----------+
           |               |    |               |
           v         appt  |    |  greeting     v
     +-----+-----+  +------+    +---+    +------+--------+
     | rag_agent  |  |              |    | cancel_handler |
     | (retrieve, |  v              v    | (reset state)  |
     |  rerank,   | +----+----+ +--+--+  +------+---------+
     |  Gemini)   | |appt_agent| |greet|         |
     +-----+------+ |(5-step   | |handler        |
           |        | form)    | +--+--+         |
           |        +----+-----+    |            |
           |             |          |            |
           v             v          v            v
         +-+-------------+----------+------------+-+
         |                   END                    |
         |          (response returned)             |
         +-----------------------------------------+
```

### Intent Classification Logic

The classifier uses a **three-pass hybrid approach**:

1. **Regex (fast, deterministic)** — Pattern matching for clear intents (high confidence)
2. **Conversation context** — If regex is uncertain and the previous turn was RAG, follow-up messages stay as RAG (avoids misclassifying short follow-ups like "how did it originate?")
3. **LLM fallback** — Only called when regex is uncertain and no context signal exists

| User Message Pattern | Detected Intent | Routed To |
|---|---|---|
| "book", "schedule", "appointment", "reserve" | `appointment` | Appointment Agent |
| "what", "how did", "why is", "explain", "?" | `rag` | RAG Agent |
| "hi", "hello", "hey", "good morning" | `greeting` | Greeting Handler |
| "cancel", "stop", "quit" (mid-appointment) | `appointment_cancel` | Cancel Handler |
| "continue booking", "resume appointment" | `appointment` | Appointment Agent |
| Follow-up to a RAG query (no appointment keywords) | `rag` | RAG Agent |
| Anything else | `rag` (default) | RAG Agent |

**Context-switching**: When a user is mid-appointment and asks a question (e.g., "What are your office hours?"), the classifier detects it's a question (not a field value) and routes to the RAG agent. The appointment state is **paused, not lost** — the user can say "continue booking" to resume.

**Follow-up detection**: Conversational follow-ups (e.g., "how did it originate?" after asking about a topic) are correctly routed to RAG by checking the previous turn's intent, preventing small LLMs from misclassifying them.

---

## RAG Pipeline

```
User Query
    |
    v
+---+-------------------+
| Query Rewriting        |  Strip filler phrases ("can you tell me...")
| (tools_service.py)     |  Collapse whitespace
+---+-------------------+
    |
    v
+---+-------------------+
| Embedding              |  bge-small-en-v1.5 (384-dim)
| (embedding_model.py)   |  Normalized L2 vectors
+---+-------------------+
    |
    v
+---+-------------------+
| Vector Search          |  Qdrant cosine similarity
| (vector_store.py)      |  Top-K candidates (default: 10)
|                        |  Optional metadata filtering
+---+-------------------+
    |
    v
+---+-------------------+
| Cross-Encoder Rerank   |  bge-reranker-base
| (reranker.py)          |  Rerank top-K -> top-N (default: 3)
+---+-------------------+
    |
    v
+---+-------------------+
| Context Assembly       |  Token budgeting (~2048 tokens)
| (context_builder.py)   |  Source attribution per chunk
+---+-------------------+
    |
    v
+---+-------------------+
| LLM Generation         |  Gemini or Ollama (auto-detected)
| (rag_agent.py)         |  System prompt + context + chat history
+------------------------+
```

---

## Appointment Booking Flow

```
User: "I want to book an appointment"
                |
                v
        +-------+--------+
        | Step 1: NAME    |  Validation: min 2 chars, must contain letters
        | "What is your   |  Error: "That doesn't look like a valid name"
        |  full name?"    |
        +-------+--------+
                |
                v
        +-------+--------+
        | Step 2: PHONE   |  Validation: Nepali mobile (starts with 98/97,
        | "What is your   |  10 digits, optional +977 prefix)
        |  phone number?" |  Auto-cleans spaces and dashes
        +-------+--------+
                |
                v
        +-------+--------+
        | Step 3: EMAIL   |  Validation: regex pattern check
        | "What is your   |  Error: "That doesn't look like a valid email"
        |  email address?"|
        +-------+--------+
                |
                v
        +-------+--------+
        | Step 4: DATE    |  Parses natural language: "next Monday",
        | "When would you |  "in 3 days", "March 25th", "tomorrow"
        |  like to        |  Converts to YYYY-MM-DD, rejects past dates
        |  schedule?"     |  Error: "I couldn't understand that date"
        +-------+--------+
                |
                v
        +-------+--------+
        | Step 5: CONFIRM |  Shows summary of all collected data
        | "Shall I        |  "yes" -> books appointment, returns ID
        |  confirm?"      |  "no"  -> cancels, resets state
        +-------+--------+
                |
         +------+------+
         |             |
    confirmed      cancelled
         |             |
         v             v
   Appointment     "Booking
   booked with      cancelled"
   unique ID
```

---

## Project Structure

```
backend/
|-- config.py                    # Centralized env config via python-dotenv
|-- logger.py                    # Centralized logging (get_logger per module)
|-- main.py                      # FastAPI app entry point, CORS middleware
|
|-- api/
|   +-- chat.py                  # REST endpoints: /chat, /appointment, /ingest, /health
|
|-- models/
|   +-- appointment_schema.py    # Pydantic schemas with strict validation
|
|-- rag/
|   |-- chunking.py              # Recursive text splitter with metadata
|   |-- embedding_model.py       # Lazy-loaded SentenceTransformer & CrossEncoder
|   |-- vector_store.py          # Qdrant abstraction (create, upsert, search)
|   |-- retriever.py             # Query embedding + top-K vector search
|   |-- reranker.py              # Cross-encoder reranking of candidates
|   |-- context_builder.py       # Token-budgeted context assembly with source tags
|   +-- ingest.py                # File/directory ingestion pipeline
|
|-- services/
|   |-- rag_service.py           # Full RAG orchestration: retrieve -> rerank -> build
|   |-- llm_provider.py          # Unified LLM abstraction (Gemini + Ollama auto-routing)
|   |-- appointment_service.py   # Booking logic, date parsing, slot management
|   |-- guardrails_service.py    # Input guardrails (profanity, injection detection)
|   |-- persistence_service.py   # JSON file-based session & appointment persistence
|   +-- tools_service.py         # Date parser, phone/email validators, query rewriter
|
+-- agents/
    |-- state.py                 # Shared AgentState TypedDict for LangGraph
    |-- intent_classifier.py     # Hybrid intent detection (regex + context + LLM fallback)
    |-- rag_agent.py             # RAG node: context retrieval + LLM call
    |-- appointment_agent.py     # Multi-step conversational booking form
    |-- greeting_handler.py      # Greeting responses
    +-- orchestrator.py          # LangGraph StateGraph: classify -> route -> respond

frontend/
|-- src/
|   |-- api/client.js            # API client for backend communication
|   |-- pages/
|   |   |-- ChatPage.jsx         # Main chat interface with typewriter effect
|   |   |-- AppointmentPage.jsx  # Appointment management page
|   |   +-- IngestPage.jsx       # Document upload page
|   |-- components/
|   |   +-- Sidebar.jsx          # Navigation sidebar
|   +-- App.jsx                  # Root app with routing
|-- Dockerfile                   # Multi-stage build (Node -> Nginx)
+-- nginx.conf                   # Reverse proxy config for API

docker-compose.yml               # Full stack: backend, frontend, Qdrant, Ollama
.env                             # Environment variable configuration
requirements.txt                 # Python dependencies
```

---

## Module Dependency Graph

```
main.py ──> api/chat.py ──> agents/orchestrator.py ──> intent_classifier.py
                |                    |                        |
                |                    +-----> rag_agent.py ----+---> services/rag_service.py
                |                    |                                    |
                |                    +-----> appointment_agent.py        +---> rag/retriever.py
                |                    |            |                      |        |
                |                    +-----> greeting_handler.py        +---> rag/reranker.py
                |                    |                                  |        |
                |                    +-----> cancel_handler             +---> rag/context_builder.py
                |                                 |
                +-----> services/appointment_service.py
                |                    |
                +-----> rag/ingest.py ──> rag/chunking.py
                                     |       |
                                     +---> rag/embedding_model.py
                                     |       |
                                     +---> rag/vector_store.py ──> Qdrant (external)
                                             |
                            services/tools_service.py ──> dateparser (external)
                                             |
                            config.py <── .env (environment variables)
                                 |
                            logger.py (used by every module)
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Backend Framework | FastAPI |
| Frontend | React + Vite (served via Nginx) |
| Agent Orchestration | LangGraph (StateGraph with conditional routing) |
| Vector Database | Qdrant (Docker container) |
| Embeddings | SentenceTransformers (`BAAI/bge-small-en-v1.5`, 384-dim) |
| Reranker | CrossEncoder (`BAAI/bge-reranker-base`) |
| LLM (Cloud) | Google Gemini 2.0 Flash API |
| LLM (Local) | Ollama (llama3.2, mistral, etc.) — auto-detected by model name |
| Input Validation | Pydantic v2 with field validators |
| Date Parsing | dateparser + parsedatetime (dual parser for natural language to YYYY-MM-DD) |
| Session Storage | In-memory dict + JSON file persistence (swap for Redis in production) |
| Containerization | Docker Compose (backend, frontend, Qdrant, Ollama) |
| Configuration | `.env` with `python-dotenv` |
| Logging | Python `logging` module (centralized, configurable level) |

---

## API Reference

All endpoints are prefixed with `/api/v1`.

### `POST /api/v1/chat`

Main conversational endpoint. Routes through the LangGraph orchestrator.

**Request:**
```json
{
  "message": "What is the refund policy?",
  "session_id": null,
  "model": "llama3.2:1b"
}
```

**Response:**
```json
{
  "reply": "Based on our documents, the refund policy states...",
  "session_id": "a1b2c3d4e5f6",
  "intent": "rag",
  "intent_confidence": "high",
  "intent_source": "regex",
  "sources": ["policy.txt"],
  "model_used": "llama3.2:1b"
}
```

### `POST /api/v1/appointment`

Direct appointment booking (bypasses conversational flow).

**Request:**
```json
{
  "name": "John Doe",
  "phone": "9812345678",
  "email": "john@example.com",
  "preferred_date": "2026-03-25",
  "reason": "General consultation"
}
```

**Response:**
```json
{
  "success": true,
  "message": "Appointment booked successfully for John Doe on 2026-03-25.",
  "appointment_id": "a1b2c3d4e5f6",
  "appointment_details": { ... }
}
```

### `POST /api/v1/ingest`

Upload a document to be chunked, embedded, and stored in Qdrant.

**Request:** Multipart form with `file` field. Query params: `chunk_size` (100-2048, default 512), `chunk_overlap` (0-256, default 64).

```bash
curl -X POST http://localhost:8000/api/v1/ingest \
  -F "file=@knowledge_base.txt"
```

**Response:**
```json
{
  "message": "Successfully ingested 'knowledge_base.txt'",
  "chunks_ingested": 42
}
```

Supported file types: `.txt`, `.md`, `.csv`, `.json`

### `GET /api/v1/health`

Health check.

```json
{ "status": "healthy" }
```

---

## Data Models

### AppointmentRequest

| Field | Type | Constraints |
|---|---|---|
| `name` | `str` | 2-100 chars, must contain letters |
| `phone` | `str` | Nepali mobile: starts with 98/97, 10 digits, optional `+977` |
| `email` | `EmailStr` | Valid email format |
| `preferred_date` | `date` | Must be today or in the future |
| `reason` | `str` (optional) | Max 500 chars |

### ChatRequest

| Field | Type | Constraints |
|---|---|---|
| `message` | `str` | 1-2000 chars |
| `session_id` | `str` (optional) | Auto-generated if absent |
| `model` | `str` (optional) | LLM model override (e.g., `"llama3.2:1b"`, `"gemini-2.0-flash"`) |

### AgentState (LangGraph)

| Field | Type | Purpose |
|---|---|---|
| `user_message` | `str` | Current user input |
| `session_id` | `str` | Conversation session ID |
| `chat_history` | `list[dict]` | Prior messages (last 20) |
| `intent` | `str` | Classified intent |
| `rag_context` | `str` | Assembled document context |
| `rag_sources` | `list[str]` | Source document names |
| `appointment_data` | `dict` | Collected appointment fields |
| `appointment_step` | `str` | Current booking step |
| `response` | `str` | Final response text |
| `error` | `str` (optional) | Error message if any |

---

## Setup & Installation

### Prerequisites

- Docker & Docker Compose
- (Optional) A Google Gemini API key ([get one here](https://aistudio.google.com/apikey)) — not needed if using Ollama only

### Option A: Docker Compose (Recommended)

Runs the full stack (backend, frontend, Qdrant, Ollama) in containers.

```bash
git clone <repo-url>
cd Context-Aware-Conversational-Agent-with-Document-Queries-Appointment-Booking-Integration

# Configure environment
cp .env.example .env
# Edit .env — set GEMINI_API_KEY or change GEMINI_MODEL_NAME to an Ollama model

# Start everything
docker compose up --build -d

# Pull an Ollama model (if using local LLM)
docker exec -it ollama ollama pull llama3.2:1b
```

- Frontend: `http://localhost:80`
- Backend API docs: `http://localhost:8000/docs`
- Qdrant dashboard: `http://localhost:6333/dashboard`

### Option B: Local Development

```bash
# Start Qdrant and Ollama via Docker
docker compose up -d qdrant ollama

# Install Python dependencies
pip install -r requirements.txt

# Install frontend dependencies
cd frontend && npm install && cd ..

# Run backend
uvicorn backend.main:app --reload

# Run frontend (separate terminal)
cd frontend && npm run dev
```

The embedding and reranker models (~100MB) are downloaded automatically on first use.

---

## Usage Examples

### Ingest documents first

```bash
# Upload a document so RAG has knowledge to search
curl -X POST http://localhost:8000/api/v1/ingest \
  -F "file=@docs/company_info.txt"
```

### Ask a question (RAG)

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What services do you offer?"}'
```

### Book an appointment (conversational)

```bash
# Start the flow
curl -s -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "I want to book an appointment"}' | jq .

# Response: "What is your full name?" + session_id
# Use the session_id for follow-up messages:

curl -s -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Jane Smith", "session_id": "<session_id>"}' | jq .

# Continue with phone, email, date, and confirmation...
```

### Context switching mid-appointment

```bash
# Mid-booking, ask a question:
curl -s -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What are your office hours?", "session_id": "<session_id>"}' | jq .

# Response: RAG answer + "You have a booking in progress. Say 'continue booking' to resume."

# Resume the appointment:
curl -s -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "continue booking", "session_id": "<session_id>"}' | jq .
```

---

## Design Decisions

| Decision | Rationale |
|---|---|
| **Layered architecture** (API -> Service -> Component) | Business logic is never in endpoints; each layer is independently testable |
| **Dual LLM backend** (Gemini + Ollama) | Cloud LLM for production quality, local Ollama for offline/free usage; provider auto-detected from model name |
| **Lazy-loaded model singletons** | Zero import-time cost; models load only when first used |
| **Three-pass intent classifier** (regex -> context -> LLM) | Regex is fast and free; conversation context catches follow-ups; LLM fallback handles edge cases |
| **Date hint detection before parsing** | Prevents dateparser from aggressively interpreting names/random text as dates |
| **Cross-encoder reranking** | Dramatically improves retrieval precision over embedding-only search |
| **Token-budgeted context** | Prevents exceeding LLM context limits regardless of chunk count |
| **Conversational appointment form** | Guides users step-by-step with clear validation feedback at each step |
| **Nepal-specific phone validation** | Enforces 98/97 prefix with 10-digit format matching real Nepali mobile numbers |
| **Context-switching with pause/resume** | Users can ask questions mid-booking without losing their progress |
| **Typewriter response animation** | Character-by-character reveal with blinking cursor creates a natural chat feel |
| **Minimum thinking delay** | Typing indicator shows for at least 600ms even for instant responses, avoiding jarring UX |
| **JSON file persistence** | Session history and appointments survive restarts without a database dependency |

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `QDRANT_URL` | `http://localhost:6333` | Qdrant server URL (use `http://qdrant:6333` in Docker) |
| `QDRANT_COLLECTION_NAME` | `documents` | Qdrant collection name |
| `GEMINI_API_KEY` | (required for Gemini) | Google Gemini API key |
| `GEMINI_MODEL_NAME` | `gemini-2.0-flash` | Default LLM model (set to an Ollama model name like `llama3.2:1b` to use local LLM) |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL (use `http://ollama:11434` in Docker) |
| `EMBEDDING_MODEL_NAME` | `BAAI/bge-small-en-v1.5` | SentenceTransformer model ID |
| `RERANKER_MODEL_NAME` | `BAAI/bge-reranker-base` | CrossEncoder reranker model ID |
| `RAG_TOP_K` | `10` | Number of candidates from vector search |
| `RAG_RERANK_TOP_N` | `3` | Number of chunks kept after reranking |
| `RAG_MAX_CONTEXT_TOKENS` | `2048` | Max token budget for assembled context |
| `LOG_LEVEL` | `INFO` | Logging level (DEBUG, INFO, WARNING, ERROR) |

---

## Frontend Features

- **Model selector** — Switch between Gemini and Ollama models from the chat header
- **Typewriter effect** — Assistant responses appear character-by-character with a blinking cursor
- **Thinking delay** — A natural pause (400-700ms) before the typing indicator appears, plus a minimum 600-1200ms indicator duration for instant responses
- **Fade-in animations** — Messages slide in smoothly when sent or received
- **Typing indicator** — Animated bouncing dots while waiting for a response
- **Session management** — "New Chat" button clears history and starts fresh
- **Meta tags** — Each response shows the detected intent, model used, and source documents
- **Document upload** — Ingest page for uploading files to the knowledge base
- **Responsive layout** — Sidebar navigation with chat, appointments, and ingest pages

---

## Troubleshooting

| Issue | Fix |
|---|---|
| **Ollama DNS failure in Docker** (`dial tcp: lookup registry.ollama.ai... connection refused`) | Add `{"dns": ["8.8.8.8", "1.1.1.1"]}` to `/etc/docker/daemon.json` and restart Docker |
| **Port 11434 already in use** | Stop host Ollama: `sudo systemctl stop ollama` before running Docker |
| **Gemini 429 quota exceeded** | Switch to Ollama: set `GEMINI_MODEL_NAME=llama3.2:1b` in `.env` |
| **`npm ci` fails during frontend build** | Run `cd frontend && npm install` to generate `package-lock.json` first |
| **Containerd blob not found** | Run `sudo docker system prune -a` to clear corrupt image cache |
