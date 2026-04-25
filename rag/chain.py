from rag.retriever import retrieve_context
from prompts.templates import REPHRASE_PROMPT, RAG_PROMPT, CASUAL_PROMPT
import logging
import re

logger = logging.getLogger(__name__)

# ── Intent Detection ──────────────────────────────────────────────────
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
    if len(cleaned.split()) <= 6 and _CASUAL_PATTERNS.match(cleaned):
        return True
    return False

def _is_summary_request(query: str) -> bool:
    """Detect if the user is asking for a general summary or detailed analysis of the whole document."""
    patterns = [
        r"tell me (everything|all) about",
        r"summarize",
        r"summary",
        r"detailed analysis",
        r"what (is|has|contains) the (document|pdf|file)",
        r"what'?s in this",
        r"overview",
        r"comprehensive"
    ]
    return any(re.search(p, query, re.IGNORECASE) for p in patterns)

def get_answer_stream(query: str, vector_store, memory, groq_client, filter_doc: str = None):
    """
    Generator that yields answer chunks for streaming.
    Yields:
        {"type": "rephrased", "content": ...}
        {"type": "sources", "content": ...}
        {"type": "chunk", "content": ...}
        {"type": "done", "answer": ..., "sources": ...}
    """
    history = memory.get_history()
    
    # ── Step 0: Intent gate ───────────────────────────────────────────
    if _is_casual(query):
        try:
            prompt = CASUAL_PROMPT.format(history=history, query=query)
            response = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
            )
            answer_text = response.choices[0].message.content.strip()
            yield {"type": "done", "answer": answer_text, "sources": []}
            memory.add_message("user", query)
            memory.add_message("assistant", answer_text, metadata={"sources": []})
            return
        except Exception as e:
            yield {"type": "error", "content": str(e)}
            return

    # ── Step 1: Rephrase query ────────────────────────────────────────
    rephrased_query = query
    if history:
        try:
            prompt = REPHRASE_PROMPT.format(history=history, query=query)
            response = groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            rephrased_query = response.choices[0].message.content.strip()
            yield {"type": "rephrased", "content": rephrased_query}
        except:
            pass
            
    # ── Step 2: Retrieve context ──────────────────────────────────────
    top_k = 15 if _is_summary_request(query) or _is_summary_request(rephrased_query) else 4
    context, sources = retrieve_context(rephrased_query, vector_store, top_k=top_k, filter_doc=filter_doc)
    yield {"type": "sources", "content": sources}
    
    # ── Step 3: Generate answer (Streaming) ───────────────────────────
    try:
        if not context or "No relevant context" in context:
            prompt = CASUAL_PROMPT.format(history=history, query=rephrased_query)
        else:
            prompt = RAG_PROMPT.format(history=history, context=context, query=rephrased_query)

        stream = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            stream=True
        )
        
        full_answer = ""
        for chunk in stream:
            content = chunk.choices[0].delta.content
            if content:
                full_answer += content
                yield {"type": "chunk", "content": content}
                
        # ── Step 4: Update memory ─────────────────────────────────────
        memory.add_message("user", query)
        memory.add_message("assistant", full_answer, metadata={"sources": sources})
        
        yield {"type": "done", "answer": full_answer, "sources": sources}
        
    except Exception as e:
        logger.error(f"Streaming Error: {e}")
        yield {"type": "error", "content": str(e)}

def get_answer(query: str, vector_store, memory, groq_client, filter_doc: str = None) -> dict:
    """Legacy wrapper for non-streaming calls."""
    # Use the generator but collect it
    gen = get_answer_stream(query, vector_store, memory, groq_client, filter_doc)
    last_answer = ""
    last_sources = []
    rephrased = query
    
    for item in gen:
        if item["type"] == "rephrased": rephrased = item["content"]
        if item["type"] == "sources": last_sources = item["content"]
        if item["type"] == "done":
            return {
                "answer": item["answer"],
                "sources": item["sources"],
                "chunks_used": len(item["sources"]),
                "rephrased_query": rephrased
            }
        if item["type"] == "error":
            raise Exception(item["content"])
            
    return {"answer": "I encountered an error generating the response.", "sources": [], "chunks_used": 0, "rephrased_query": rephrased}
