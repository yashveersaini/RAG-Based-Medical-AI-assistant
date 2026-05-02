from src.retriever import get_hybrid_retriever
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from config import settings


# Medical prompt template (your exact template) 

MEDICAL_TEMPLATE = """
You are a medical expert assistant.

Use ONLY the information provided in the context below to answer the question.
If the answer is not clearly present, say "I don't have enough information to answer this."

Context:
{context}

Question:
{query}

Instructions:
- Combine information from multiple documents if needed
- Avoid repetition
- Provide a structured answer where appropriate (Definition, Causes, Symptoms, Treatment)
- Be precise and factual
- Do not add information not present in the context
"""

prompt = PromptTemplate(
    template=MEDICAL_TEMPLATE,
    input_variables=["context", "query"]
)

# Format retrieved docs into context string 

def format_docs(docs) -> tuple[str, list]:
    """
    Returns (context_string, sources_list)
    context_string → goes into the prompt
    sources_list   → shown in UI as citations
    """
    context_parts = []
    sources = []

    for i, doc in enumerate(docs):
        score = doc.metadata.get("relevance_score", 0)
        source = doc.metadata.get("source", "Medical Book")
        page = doc.metadata.get("page", 0)

        context_parts.append(
            f"Document {i+1}:\n{doc.page_content}"
        )

        sources.append({
            "source": source,
            "page": int(page),
            "preview": doc.page_content[:120] + "...",
            "score": round(float(score), 3) if score else None
        })

    return "\n\n".join(context_parts), sources


# Main query function 

def get_answer(query: str, embeddings) -> dict:
    """
    Your exact pipeline:
    1. retrieve docs
    2. format context
    3. run prompt | model chain
    4. return answer + sources
    """
    # retrieve
    docs = get_hybrid_retriever(query, embeddings)

    # format context
    context, sources = format_docs(docs)

    # build and run chain
    model = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=settings.gemini_api_key
    )

    chain = prompt | model
    response = chain.invoke({"context": context, "query": query})

    return {
        "answer": response.content,
        "sources": sources
    }
