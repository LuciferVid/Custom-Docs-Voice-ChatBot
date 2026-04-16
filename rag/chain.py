from rag.retriever import retrieve_context
from prompts.templates import REPHRASE_PROMPT, RAG_PROMPT
import logging

logger = logging.getLogger(__name__)

def get_answer(query: str, vector_store, memory, gemini_client, filter_doc: str = None) -> dict:
    """
    Orchestrates the RAG process using Google Gemini.
    Pure Gemini version for high-performance retrieval.
    """
    history = memory.get_history()
    
    # Step 1: Rephrase query if history exists
    rephrased_query = query
    if history:
        try:
            prompt = REPHRASE_PROMPT.format(history=history, query=query)
            response = gemini_client.models.generate_content(
                model="gemini-1.5-flash",
                contents=prompt,
                config={"temperature": 0}
            )
            rephrased_query = response.text.strip()
            logger.info(f"Rephrased query: {rephrased_query}")
        except Exception as e:
            logger.error(f"Error rephrasing query with Gemini: {e}")
            
    # Step 2: Retrieve context
    context, sources = retrieve_context(rephrased_query, vector_store, filter_doc=filter_doc)
    
    # Step 3: Generate answer
    try:
        prompt = RAG_PROMPT.format(history=history, context=context, query=rephrased_query)
        response = gemini_client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
            config={"temperature": 0}
        )
        answer_text = response.text.strip()
    except Exception as e:
        logger.error(f"Error generating answer with Gemini: {e}")
        if not context or "No relevant context" in context:
            answer_text = "Your document context seems to have expired (likely due to a server refresh). Please re-upload your document to continue our analysis."
        else:
            answer_text = "I'm having trouble connecting to the AI brain right now. Please try your question again in a moment."

    # Step 4: Update memory
    memory.add_message("user", query)
    memory.add_message("assistant", answer_text, metadata={"sources": sources})
    
    return {
        "answer": answer_text,
        "sources": sources,
        "chunks_used": len(sources),
        "rephrased_query": rephrased_query
    }
