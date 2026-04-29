import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.evaluation import generate_test_dataset
from config import settings

def prepare_and_generate():
    # 1. Load your Medical Book (Update the path to your actual PDF)
    pdf_path = "data/Medical_book.pdf" 
    if not os.path.exists(pdf_path):
        print(f"Error: {pdf_path} not found. Put your PDF in the data folder.")
        return

    loader = PyMuPDFLoader(pdf_path)
    docs = loader.load()

    # 2. Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(docs)

    print(f"Loaded {len(chunks)} chunks. Generating 100 test questions...")

    # 3. Call the generation function from your evaluation script
    # This will create 'data/test_dataset.json'
    generate_test_dataset(chunks, num_questions=100)

if __name__ == "__main__":
    prepare_and_generate()