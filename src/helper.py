from langchain_community.document_loaders import PyMuPDFLoader, DirectoryLoader
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter   # fixed import
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from config import settings
from typing import List


def load_pdf_files(data: str) -> List[Document]:
    loader = DirectoryLoader(
        data,
        glob="*.pdf",
        loader_cls=PyMuPDFLoader
    )
    return loader.load()


def enrich_metadata(docs: List[Document]) -> List[Document]:
    """
        Keep source + add page number and filename separately.
    """
    enriched = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        filename = source.split("/")[-1]          
        enriched.append(
            Document(
                page_content=doc.page_content,
                metadata={
                    "source": source,
                    "filename": filename,
                    "page": doc.metadata.get("page", 0),  # PyMuPDF provides this
                }
            )
        )
    return enriched


def text_split(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,                         
        length_function=len,
        separators=["\n\n", "\n", ".", " "]       # added: smarter splitting
    )
    chunks = splitter.split_documents(docs)

    # tag each chunk with its position — needed for citations later
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i

    return chunks


def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=settings.embed_model)