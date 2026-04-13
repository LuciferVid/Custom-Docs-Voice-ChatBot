from rag.retriever import retrieve_context
from prompts.templates import REPHRASE_PROMPT, RAG_PROMPT
import logging

logger = logging.getLogger(__name__)

def get_answer(query: str, vector_store, memory, openai_client, filter_doc: str = None) -> dict:
    """
    Orchestrates the RAG process: rephrasing, retrieval, and generation.
    """
    history = memory.get_history()
    
    # Step 1: Rephrase query if history exists
    rephrased_query = query
    if history:
        try:
            response = openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that rephrases questions."},
                    {"role": "user", "content": REPHRASE_PROMPT.format(history=history, query=query)}
                ],
                temperature=0
            )
            rephrased_query = response.choices[0].message.content.strip()
            logger.info(f"Rephrased query: {rephrased_query}")
        except Exception as e:
            logger.error(f"Error rephrasing query: {e}")
            
    # Step 2: Retrieve context
    context, sources = retrieve_context(rephrased_query, vector_store, filter_doc=filter_doc)
    
    # Step 3: Generate answer
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful document assistant."},
                {"role": "user", "content": RAG_PROMPT.format(history=history, context=context, query=rephrased_query)}
            ],
            temperature=0
        )
        answer_text = response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Error generating answer: {e}")
        answer_text = "I encountered an error while searching for the answer."

    # Step 4: Update memory
    memory.add_message("user", query)
    memory.add_message("assistant", answer_text, metadata={"sources": sources})
    
    return {
        "answer": answer_text,
        "sources": sources,
        "chunks_used": len(sources),
        "rephrased_query": rephrased_query
    }
