from rag.retriever import retrieve_context
from prompts.templates import REPHRASE_PROMPT, RAG_PROMPT, CASUAL_PROMPT
import logging
import re

logger = logging.getLogger(__name__)

# ── Intent Detection ──────────────────────────────────────────────────
# Fast keyword-based classifier to avoid wasting a retrieval call on greetings.

_CASUAL_PATTERNS = re.compile(
    r"^\s*("
    r"h(i|ey|ello|owdy|ola)"
    r"|yo\b"
    r"|sup\b"
    r"|what'?s\s*up"
    r"|how\s*(are|r)\s*(you|u|ya)"
    r"|good\s*(morning|afternoon|evening|night)"
    r"|gm\b|gn\b"
    r"|thank(s| you)"
    r"|bye|goodbye|see\s*ya|later"
    r"|ok(ay)?"
    r"|yes|yeah|yep|yup|no|nah|nope"
    r"|cool|nice|great|awesome|wow"
    r"|lol|lmao|haha"
    r"|i\s*got\s*it"
    r"|who\s*are\s*you"
    r"|what\s*can\s*you\s*do"
    r"|help\b"
    r")\s*[?!.,]*\s*$",
    re.IGNORECASE,
)


def _is_casual(query: str) -> bool:
    """Return True if the query is clearly conversational / not document-related."""
    cleaned = query.strip()
    # Very short messages (<6 words) that match casual patterns
    if len(cleaned.split()) <= 6 and _CASUAL_PATTERNS.match(cleaned):
        return True
    return False


def get_answer(query: str, vector_store, memory, gemini_client, filter_doc: str = None) -> dict:
    """
    Orchestrates the RAG process using Gemini 1.5 Flash (High Limit Free Tier).
    """
    history = memory.get_history()
    sources = []

    def call_gemini(prompt):
        try:
            response = gemini_client.models.generate_content(
                model="gemini-1.5-flash",
                contents=prompt
            )
            if not response or not hasattr(response, 'text'):
                 raise ValueError("Gemini returned an empty response or was blocked by safety filters.")
            return response.text.strip()
        except Exception as e:
            logger.error(f"Gemini API Error: {e}")
            raise e

    # ── Step 0: Intent gate ───────────────────────────────────────────
    if _is_casual(query):
        logger.info(f"Casual intent detected — skipping retrieval")
        try:
            prompt = CASUAL_PROMPT.format(history=history, query=query)
            answer_text = call_gemini(prompt)
        except Exception as e:
            answer_text = "Hey! I'm here and ready to help. Ask me anything about your documents! 👋"

        memory.add_message("user", query)
        memory.add_message("assistant", answer_text, metadata={"sources": []})
        return {
            "answer": answer_text,
            "sources": [],
            "chunks_used": 0,
            "rephrased_query": query,
        }

    # ── Step 1: Rephrase query ────────────────────────────────────────
    rephrased_query = query
    if history:
        try:
            prompt = REPHRASE_PROMPT.format(history=history, query=query)
            rephrased_query = call_gemini(prompt)
            logger.info(f"Rephrased query: {rephrased_query}")
        except:
            pass # Keep original query if rephrasing fails
            
    # ── Step 2: Retrieve context ──────────────────────────────────────
    context, sources = retrieve_context(rephrased_query, vector_store, filter_doc=filter_doc)
    
    # ── Step 3: Generate answer ───────────────────────────────────────
    try:
        if not context or "No relevant context" in context:
            prompt = CASUAL_PROMPT.format(history=history, query=rephrased_query)
        else:
            prompt = RAG_PROMPT.format(history=history, context=context, query=rephrased_query)

        answer_text = call_gemini(prompt)
    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg or "QUOTA" in error_msg.upper():
            answer_text = "⚠️ **Gemini Limit Reached**: Please try again in a few minutes."
        elif "API_KEY_INVALID" in error_msg:
            answer_text = "⚠️ **Invalid API Key**: Please check your GEMINI_API_KEY in the environment settings."
        else:
            answer_text = f"⚠️ **AI Brain Connection Error**: {error_msg}"


    # ── Step 4: Update memory ─────────────────────────────────────────
    memory.add_message("user", query)
    memory.add_message("assistant", answer_text, metadata={"sources": sources})
    
    return {
        "answer": answer_text,
        "sources": sources,
        "chunks_used": len(sources),
        "rephrased_query": rephrased_query
    }



