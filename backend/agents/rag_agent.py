"""
RAG agent node for LangGraph.

Retrieves relevant document context and generates an LLM-powered answer.
"""

from __future__ import annotations

from google import genai

from backend.agents.state import AgentState
from backend.config import GEMINI_API_KEY, GEMINI_MODEL_NAME
from backend.logger import get_logger
from backend.services.rag_service import answer_from_documents

logger = get_logger(__name__)

_RAG_SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on the provided context.
Rules:
- Answer ONLY based on the context below. Do not use outside knowledge.
- If the context does not contain enough information, say so clearly.
- Be concise and direct.
- Cite the source when possible.

Context:
{context}
"""

_gemini_client = genai.Client(api_key=GEMINI_API_KEY)


def _call_gemini(prompt: str, system_instruction: str, model: str = GEMINI_MODEL_NAME) -> str:
    """Send a prompt to Gemini and return the response text."""
    logger.info("Calling Gemini model: %s", model)
    response = _gemini_client.models.generate_content(
        model=model,
        contents=prompt,
        config=genai.types.GenerateContentConfig(
            system_instruction=system_instruction,
        ),
    )
    return response.text


def handle_rag_query(state: AgentState) -> AgentState:
    """
    LangGraph node: answer the user's question from documents.

    Steps:
        1. Run the RAG service to get context.
        2. Build an LLM prompt with the context.
        3. Call the LLM to generate an answer.
        4. Write the answer into ``state["response"]``.
    """
    query = state.get("user_message", "")
    logger.info("RAG agent processing query: %.80s", query)

    try:
        rag_result = answer_from_documents(query)

        if not rag_result["context"]:
            state["response"] = (
                "I couldn't find any relevant information in the documents "
                "to answer your question. Could you rephrase or ask something else?"
            )
            state["rag_sources"] = []
            return state

        state["rag_context"] = rag_result["context"]
        state["rag_sources"] = rag_result["sources"]

        system_prompt = _RAG_SYSTEM_PROMPT.format(context=rag_result["context"])

        # Include recent chat history for conversational context
        chat_history = state.get("chat_history", [])
        history_text = ""
        if chat_history:
            recent = chat_history[-6:]  # last 3 exchanges
            history_text = "\n".join(
                f"{msg['role'].capitalize()}: {msg['content']}" for msg in recent
            )
            history_text = f"\nConversation history:\n{history_text}\n"

        full_prompt = f"{history_text}User question: {query}"

        selected_model = state.get("llm_model", GEMINI_MODEL_NAME)
        answer = _call_gemini(prompt=full_prompt, system_instruction=system_prompt, model=selected_model)

        # If there's a paused appointment flow, remind the user
        paused_step = state.get("appointment_step", "")
        if paused_step and paused_step != "complete":
            answer += (
                "\n\n---\n"
                "_You also have an appointment booking in progress. "
                "Just say **\"continue booking\"** to resume, "
                "or **\"cancel booking\"** to discard it._"
            )

        state["response"] = answer
        logger.info("RAG agent generated answer (%d chars)", len(answer))

    except Exception as exc:
        logger.exception("RAG agent failed")
        state["error"] = str(exc)
        state["response"] = (
            "I'm sorry, I encountered an error while searching the documents. "
            "Please try again."
        )

    return state
