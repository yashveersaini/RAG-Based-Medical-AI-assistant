system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrived context to answer "
    "The question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

# prompt.py

def qa_generation_prompt(context: str) -> str:
    return f"""
        You are a GenAI engineer creating a high-quality RAG evaluation dataset from medical documents.
        Your goal is to simulate realistic human-generated QA pairs grounded strictly in source text.

        Given the text:
        1. Write one natural question (not copied).
        2. Write a short, correct answer (1–2 sentences, only from text).
        3. Extract supporting context.

        Rules for context:
        - Copy from text (no paraphrasing)
        - Remove unrelated parts
        - Clean formatting/OCR issues
        - Max 1–3 sentences

        Text:
        {context}

        Reply ONLY in JSON format:
        {{
        "question": "...",
        "answer": "...",
        "context": "..."
        }}
    """