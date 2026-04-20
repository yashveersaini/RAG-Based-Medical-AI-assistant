from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from src.helper import get_embeddings
from src.prompt import system_prompt
from config import settings


def build_rag_chain():
    """
    Builds and returns the RAG chain.
    Called once at startup in main.py.
    """
    embeddings = get_embeddings()

    docsearch = PineconeVectorStore.from_existing_index(
        index_name=settings.pinecone_index_name,
        embedding=embeddings
    )

    retriever = docsearch.as_retriever(
        search_type="similarity",
        search_kwargs={"k": settings.top_k}
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=settings.gemini_api_key    # fixed: correct key now
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    return rag_chain, docsearch


async def run_query(rag_chain, query: str) -> dict:
    """
    Runs the RAG chain and returns answer + source citations.
    """
    response = await rag_chain.ainvoke({"input": query})  # async call

    # extract citations from retrieved context docs
    sources = []
    for doc in response.get("context", []):
        sources.append({
            "filename": doc.metadata.get("filename", "unknown"),
            "page": doc.metadata.get("page", 0),
            "preview": doc.page_content[:150] + "..."
        })

    return {
        "answer": response["answer"],
        "sources": sources
    }