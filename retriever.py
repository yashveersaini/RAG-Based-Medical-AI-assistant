import os
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from config import settings

# Setup Environment
os.environ["PINECONE_API_KEY"] = settings.pinecone_api_key
os.environ["COHERE_API_KEY"] = settings.cohere_api_key

def get_hybrid_retriever(query: str):
    """
    Standardizes retrieval for your specific Pinecone metadata:
    {chunk_index: int, page: int, source: str}
    """
    # 1. Initialize Embeddings
    embeddings = HuggingFaceEmbeddings(model_name=settings.embed_model)
    
    # 2. Setup Vector Store Retriever
    vectorstore = PineconeVectorStore(
        index_name=settings.pinecone_index_name, 
        embedding=embeddings
    )
    vector_retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 20})

    # 3. Fetch Candidate Docs for Keyword Search
    # BM25 needs actual text to build its frequency table.
    candidate_docs = vector_retriever.invoke(query)
    
    if not candidate_docs:
        return []

    # 4. Initialize BM25 (Keyword Path)
    bm25_retriever = BM25Retriever.from_documents(candidate_docs)
    bm25_retriever.k = 10
    
    # 5. Combine into Ensemble (RRF Fusion)
    ensemble_retriever = EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=[0.5, 0.5]
    )

    # 6. Final Rerank with Cohere
    # This uses the 'rerank-english-v3.0' model to pick the absolute top 5
    compressor = CohereRerank(model="rerank-english-v3.0", top_n=5)
    
    hybrid_pipeline = ContextualCompressionRetriever(
        base_compressor=compressor, 
        base_retriever=ensemble_retriever
    )

    return hybrid_pipeline.invoke(query)

