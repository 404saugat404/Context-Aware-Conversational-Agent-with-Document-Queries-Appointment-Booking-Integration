# Context-Aware Conversational Agent

A production-ready conversational chatbot that seamlessly handles **document-based Q&A (RAG)** and **appointment booking** through a unified chat interface. Built with FastAPI, LangGraph, Qdrant, and Gemini.

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
              | (regex pattern matching)    |  |    | upsert into Qdrant    |
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
    | Gemini  |               |          |        |
    | LLM API |               |          |        |
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
                    | (pattern-based)   |
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

| User Message Pattern | Detected Intent | Routed To |
|---|---|---|
| "book", "schedule", "appointment", "reserve" | `appointment` | Appointment Agent |
| "what", "how", "why", "explain", "?" | `rag` | RAG Agent |
| "hi", "hello", "hey", "good morning" | `greeting` | Greeting Handler |
| "cancel", "stop", "quit" (mid-appointment) | `appointment_cancel` | Cancel Handler |
| "continue booking", "resume appointment" | `appointment` | Appointment Agent |
| Anything else | `rag` (default) | RAG Agent |

**Context-switching**: When a user is mid-appointment and asks a question (e.g., "What are your office hours?"), the classifier detects it's a question (not a field value) and routes to the RAG agent. The appointment state is **paused, not lost** -- the user can say "continue booking" to resume.

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
| LLM Generation         |  Gemini 1.5 Flash
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
        | Step 2: PHONE   |  Validation: 10-15 digits, optional leading +
        | "What is your   |  Auto-cleans spaces and dashes
        |  phone number?" |  Error: "That phone number doesn't look right"
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
|   |-- appointment_service.py   # Booking logic, date parsing, slot management
|   +-- tools_service.py         # Date parser, phone/email validators, query rewriter
|
+-- agents/
    |-- state.py                 # Shared AgentState TypedDict for LangGraph
    |-- intent_classifier.py     # Pattern-based intent detection with context-switching
    |-- rag_agent.py             # RAG node: context retrieval + Gemini LLM call
    |-- appointment_agent.py     # Multi-step conversational booking form
    |-- greeting_handler.py      # Greeting responses
    +-- orchestrator.py          # LangGraph StateGraph: classify -> route -> respond

docker/
+-- docker-compose.yml           # Qdrant + Ollama containers

.env.example                     # Environment variable template
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
| Agent Orchestration | LangGraph (StateGraph with conditional routing) |
| Vector Database | Qdrant (Docker container) |
| Embeddings | SentenceTransformers (`BAAI/bge-small-en-v1.5`, 384-dim) |
| Reranker | CrossEncoder (`BAAI/bge-reranker-base`) |
| LLM | Google Gemini 1.5 Flash API |
| Input Validation | Pydantic v2 with field validators |
| Date Parsing | dateparser (natural language to YYYY-MM-DD) |
| Session Storage | In-memory dict (swap for Redis in production) |
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
  "session_id": null
}
```

**Response:**
```json
{
  "reply": "Based on our documents, the refund policy states...",
  "session_id": "a1b2c3d4e5f6",
  "intent": "rag",
  "sources": ["policy.txt"]
}
```

### `POST /api/v1/appointment`

Direct appointment booking (bypasses conversational flow).

**Request:**
```json
{
  "name": "John Doe",
  "phone": "9876543210",
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
| `phone` | `str` | 10-15 digits, optional leading `+` |
| `email` | `EmailStr` | Valid email format |
| `preferred_date` | `date` | Must be today or in the future |
| `reason` | `str` (optional) | Max 500 chars |

### ChatRequest

| Field | Type | Constraints |
|---|---|---|
| `message` | `str` | 1-2000 chars |
| `session_id` | `str` (optional) | Auto-generated if absent |

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

- Python 3.10+
- Docker & Docker Compose
- A Google Gemini API key ([get one here](https://aistudio.google.com/apikey))

### 1. Clone and configure

```bash
git clone <repo-url>
cd Context-Aware-Conversational-Agent-with-Document-Queries-Appointment-Booking-Integration
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY
```

### 2. Start Qdrant

```bash
cd docker
docker compose up -d
cd ..
```

Verify at `http://localhost:6333/dashboard`.

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

The embedding and reranker models (~100MB) are downloaded automatically on first use.

### 4. Run the server

```bash
uvicorn backend.main:app --reload
```

API docs available at `http://localhost:8000/docs`.

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
| **Lazy-loaded model singletons** | Zero import-time cost; models load only when first used |
| **Regex-based intent classifier** | Deterministic, fast, no LLM cost; sufficient for well-defined intents |
| **Cross-encoder reranking** | Dramatically improves retrieval precision over embedding-only search |
| **Token-budgeted context** | Prevents exceeding LLM context limits regardless of chunk count |
| **Conversational appointment form** | Guides users step-by-step with clear validation feedback at each step |
| **Context-switching with pause/resume** | Users can ask questions mid-booking without losing their progress |
| **In-memory session store** | Simple for development; designed to be swapped for Redis with no code changes |

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `QDRANT_URL` | `http://localhost:6333` | Qdrant server URL |
| `QDRANT_COLLECTION_NAME` | `documents` | Qdrant collection name |
| `GEMINI_API_KEY` | (required) | Google Gemini API key |
| `EMBEDDING_MODEL_NAME` | `BAAI/bge-small-en-v1.5` | SentenceTransformer model ID |
| `RERANKER_MODEL_NAME` | `BAAI/bge-reranker-base` | CrossEncoder reranker model ID |
| `RAG_TOP_K` | `10` | Number of candidates from vector search |
| `RAG_RERANK_TOP_N` | `3` | Number of chunks kept after reranking |
| `RAG_MAX_CONTEXT_TOKENS` | `2048` | Max token budget for assembled context |
| `LOG_LEVEL` | `INFO` | Logging level (DEBUG, INFO, WARNING, ERROR) |
