REPHRASE_PROMPT = """
Given conversation history and new question, rewrite the question 
to be standalone (resolve pronouns, add context).
History: {history}
Question: {query}
Standalone question:
"""

RAG_PROMPT = """
You are a helpful document assistant. Answer questions using ONLY 
the context provided. Follow these rules strictly:

1. Use ONLY information from the context below
2. If answer not in context: say "I couldn't find this in your documents."
3. Cite sources: end with "📄 Source: [filename, Page X]"
4. Be clear and concise
5. For follow-ups, use conversation history for context

Conversation History:
{history}

Retrieved Context:
{context}

Question: {query}

Answer:
"""

SUGGEST_PROMPT = """
You are a document analyzer. Based on the following document excerpts, generate 3 professional and insightful questions 
that a user would want to ask to understand the document's core contents better.
Rules:
1. Questions must be specific to the content.
2. Questions must be insightful (not just "what is this?").
3. Return ONLY a JSON list of strings.

Context:
{context}

JSON Output:
"""
