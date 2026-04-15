from rag.retriever import retrieve_context
from prompts.templates import REPHRASE_PROMPT, RAG_PROMPT
import logging

logger = logging.getLogger(__name__)

def get_answer(query: str, vector_store, memory, openai_client, gemini_client=None, provider: str = "openai", filter_doc: str = None) -> dict:
    """
    Orchestrates the RAG process: rephrasing, retrieval, and generation.
    Supports either OpenAI or Google Gemini.
    """
    history = memory.get_history()
    
    # Step 1: Rephrase query if history exists
    rephrased_query = query
    if history:
        try:
            prompt = REPHRASE_PROMPT.format(history=history, query=query)
            # Force Gemini if OpenAI client is missing
            actual_provider = provider
            if actual_provider == "openai" and not openai_client:
                actual_provider = "gemini"

            if actual_provider == "gemini" and gemini_client:
                response = gemini_client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=prompt,
                    config={"temperature": 0}
                )
                rephrased_query = response.text.strip()
            elif openai_client:
                response = openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that rephrases questions."},
                        {"role": "user", "content": prompt}
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
        prompt = RAG_PROMPT.format(history=history, context=context, query=rephrased_query)
        # Force Gemini if OpenAI client is missing
        actual_provider = provider
        if actual_provider == "openai" and not openai_client:
            actual_provider = "gemini"

        if actual_provider == "gemini" and gemini_client:
            response = gemini_client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt,
                config={"temperature": 0}
            )
            answer_text = response.text.strip()
        elif openai_client:
            response = openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a helpful document assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0
            )
            answer_text = response.choices[0].message.content.strip()
        else:
            answer_text = "No AI provider configured. Please provide a Gemini or OpenAI API key."
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
