import os
import json
from pathlib import Path
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

# Import evaluation logic from your src folder
from src.evaluation import (
    run_evaluation_pipeline, 
    evaluate_ragas, 
    save_eval_history, 
    print_history
)
from src.retriever import get_hybrid_retriever
from config import settings

def get_rag_chain():
    """
    Constructs the RAG chain using the prompt template 
    and Gemini model as per your pipeline steps.
    """
    template = """
    You are a medical expert assistant.

    Use ONLY the information provided in the context below to answer the question.
    If the answer is not clearly present, say "I don't know".

    Context:
    {context}

    Question:
    {query}

    Instructions:
    - Combine information from multiple documents
    - Avoid repetition
    - Provide a structured answer (Definition, Causes, Symptoms, Treatment)
    - Be precise and factual
    """
    
    prompt = PromptTemplate(
        template=template, 
        input_variables=['context', 'query']
    )

    # Use the model version stable in 2026 
    model = ChatGoogleGenerativeAI(
        model='gemini-2.5-flash', 
        google_api_key=settings.gemini_api_key
    )

    # This creates the LCEL chain
    return prompt | model

def main():
    # 1. Load Test Data
    data_path = Path("data/test_dataset.json")
    if not data_path.exists():
        print("Error: data/test_dataset.json not found! Run generation first.")
        return
        
    with open(data_path, "r") as f:
        test_data = json.load(f)

    # 2. Initialize Components
    print("Initializing components...")
    
    # We pass a dummy query to initialize the retriever if needed
    # (Your get_hybrid_retriever usually takes a query string)
    retriever = get_hybrid_retriever
    
    # Define the chain
    chain = get_rag_chain()

    # 3. Run Evaluation Pipeline
    # This calls the LLM and Retriever for every question in your JSON
    print(f"Running pipeline on {len(test_data)} questions...")

    result_path = "data/retriever_result.json"

    if os.path.exists(result_path):
        print("Loading existing results from file...")
        with open(result_path, "r", encoding="utf-8") as f:
            results = json.load(f) 
    else:
        print("No cached results found. Running pipeline...")
        results = run_evaluation_pipeline(test_data, retriever, chain)

        # Save results
        os.makedirs("data", exist_ok=True)
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)


    # 4. Score with RAGAS
    print("Calculating RAGAS metrics...")
    scores = evaluate_ragas(results)

    # 5. Save and Compare
    save_eval_history(scores, label="Hybrid-BM25-Rerank-v1")
    print_history()

if __name__ == "__main__":
    main()