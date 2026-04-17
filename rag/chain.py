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
                model="gemini-flash-latest",
                contents=prompt,
                config={"temperature": 0}
            )
            rephrased_query = response.text.strip()
            logger.info(f"Rephrased query: {rephrased_query}")
        except Exception as e:
            logger.error(f"Error rephrasing query with Gemini: {e}")
            
    # Step 2: Retrieve context
    logger.info(f"Retrieving context for query: {rephrased_query}")
    context, sources = retrieve_context(rephrased_query, vector_store, filter_doc=filter_doc)
    
    # Context Diagnostic
    if context and "No relevant context" not in context:
        logger.info(f"Retrieved {len(sources)} sources. Context snippet: {context[:200]}...")
    else:
        logger.warning("No relevant context found in the brain.")
    
    # Step 3: Generate answer
    try:
        # If no relevant context found, but vector store exists
        if not context or "No relevant context" in context:
            prompt = f"""You are 'Intelligence Core', an advanced AI assistant. 
The user asked: "{rephrased_query}"

If this is a casual greeting or general conversation, respond naturally and helpfully without mentioning documents.
If it is a factual question, answer using your general knowledge, but politely note that the specific detail was not found in the uploaded documents."""
        else:
            prompt = RAG_PROMPT.format(history=history, context=context, query=rephrased_query)

        response = gemini_client.models.generate_content(
            model="gemini-flash-latest",
            contents=prompt,
            config={"temperature": 0}
        )
        answer_text = response.text.strip()
    except Exception as e:
        logger.error(f"Error generating answer with Gemini: {e}")
        # Identify if this is a context loss issue
        if "No intelligence context" in str(e) or "empty index" in str(e):
             raise Exception("Intelligence context lost. Please re-sync.")
        answer_text = "I'm having trouble connecting to the AI brain right now. Please try again in a moment."

    # Step 4: Update memory
    memory.add_message("user", query)
    memory.add_message("assistant", answer_text, metadata={"sources": sources})
    
    return {
        "answer": answer_text,
        "sources": sources,
        "chunks_used": len(sources),
        "rephrased_query": rephrased_query
    }
