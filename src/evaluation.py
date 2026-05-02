import json
import os
from datetime import datetime
from pathlib import Path
from typing import List
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas.cache import DiskCacheBackend
from ragas.llms import LangchainLLMWrapper
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from config import settings
from src.prompt import qa_generation_prompt
import time  
import random
from dotenv import load_dotenv

load_dotenv()


# ── 1. Generate Test Dataset ──────────────────────────────────────

def generate_test_dataset(chunks: List[Document], num_questions: int = 100) -> List[dict]:
    """
    Generates Q&A pairs and saves to JSON after EVERY successful generation.
    """
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=settings.gemini_api_key
    )

    sampled_chunks = random.sample(chunks, min(num_questions, len(chunks)))
    
    file_path = Path("data/test_dataset.json")
    file_path.parent.mkdir(exist_ok=True)

    # 1. Load existing data if the script was stopped previously
    test_data = []
    if file_path.exists():
        try:
            with open(file_path, "r") as f:
                test_data = json.load(f)
            print(f"Resuming: Found {len(test_data)} existing questions.")
        except json.JSONDecodeError:
            test_data = []

    # Calculate how many more we need
    remaining_to_generate = num_questions - len(test_data)
    if remaining_to_generate <= 0:
        print("Dataset already complete.")
        return test_data

    # Only loop for the remaining number needed
    for i in range(remaining_to_generate):
        current_chunk = sampled_chunks[i + len(test_data)]
        print(f"Generating question {len(test_data) + 1}/{num_questions}...")

        prompt = qa_generation_prompt(current_chunk.page_content)

        try:
            response = llm.invoke(prompt)
            content = response.content.replace("```json", "").replace("```", "").strip()
            qa = json.loads(content)

            # 2. Add to list
            test_data.append({
                "question": qa.get("question", ""),
                "ground_truth": qa.get("answer", ""),
                "reference_context": qa.get("context", "")
            })

            # 3. SAVE HAND-TO-HAND (Overwrite the file immediately)
            with open(file_path, "w") as f:
                json.dump(test_data, f, indent=4)
            
            time.sleep(5)  # Sleep after each successful generation to be safe

        except Exception as e:
            if "429" in str(e):
                print("Rate limit hit! Sleeping for 60 seconds...")
                time.sleep(60)
            else:
                print(f"Error at step {i}: {e}")
            continue

        # 4. SLEEP to respect 5 RPM limit (13 seconds is safer than 12)
        if i < remaining_to_generate - 1:
            time.sleep(13)

    print(f"Done! Total questions saved: {len(test_data)}")

    return test_data


# ── 2. Run Pipeline & Collect Results ─────────────────────────────

def run_evaluation_pipeline(test_data: List[dict], retriever, chain) -> List[dict]:
    """Runs the actual RAG pipeline to get answers for evaluation with resume support."""
    
    os.makedirs('data', exist_ok=True)
    file_path = 'data/retriever_result.json'
    
    # 1. Load existing results if file exists
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
        print(f"Resuming from {len(results)} already processed items...")
    else:
        results = []

    start_index = len(results)

    embeddings = HuggingFaceEmbeddings(model_name=settings.embed_model)

    # 2. Continue loop from where it stopped
    for i in range(start_index, len(test_data)):
        item = test_data[i]
        print(f"Processing question {i+1}/{len(test_data)}")

        try:
            # Retrieve
            docs = retriever(item["question"], embeddings)
            contexts = [doc.page_content for doc in docs]

            # Format context
            formatted_context = "\n\n".join([f"Doc {j+1}: {c}" for j, c in enumerate(contexts)])

            # Generate answer
            response = chain.invoke({"context": formatted_context, "query": item["question"]})
            time.sleep(20)  # to handle limit of gemini model

            result = {
                "question": item["question"],
                "answer": response.content,
                "contexts": contexts,
                "ground_truth": item["ground_truth"]
            }

            results.append(result)

            # 3. Save after each iteration (IMPORTANT)
            with open(file_path, 'w', encoding='utf-8') as file:
                json.dump(results, file, indent=4, ensure_ascii=False)

        except Exception as e:
            print(f"Error at index {i}: {e}")
            break  

    return results


# ── 3. Score with RAGAS ───────────────────────────────────────────


def evaluate_ragas(results: list[dict]) -> dict:
    dataset = Dataset.from_list(results)
    
    # Initialize the Disk Cache
    # This creates a '.cache' folder in your project directory
    cacher = DiskCacheBackend() 
    
    # Wrap your LLM with the cacher
    base_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    llm = LangchainLLMWrapper(base_llm, cache=cacher)
    
    embeddings = HuggingFaceEmbeddings(model_name=settings.embed_model)

    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=llm,
        embeddings=embeddings,
        batch_size=2 # Keep this low for Free Tier
    )
    
    return {k: round(float(v), 4) for k, v in result.items()}


# ── 4. JSON-Based Result Saving (Replaces SQLite) ─────────────────

def save_eval_history(scores: dict, label: str = "baseline"):
    """Appends scores to a JSON file history."""
    file_path = Path("data/eval_history.json")
    
    new_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "label": label,
        "scores": scores
    }
    
    history = []
    if file_path.exists():
        with open(file_path, "r") as f:
            history = json.load(f)
            
    history.append(new_entry)
    
    with open(file_path, "w") as f:
        json.dump(history, f, indent=4)
        
    print(f"Successfully saved evaluation history with label: {label}")

def print_history():
    """Prints past evaluation results in a readable format."""
    file_path = Path("data/eval_history.json")
    if not file_path.exists():
        print("No history found.")
        return

    with open(file_path, "r") as f:
        history = json.load(f)

    print(f"\n{'Date':<20} {'Label':<15} {'Faith':<8} {'Relv':<8} {'Prec':<8} {'Recall':<8}")
    print("-" * 75)
    for run in history:
        s = run["scores"]
        print(f"{run['timestamp']:<20} {run['label']:<15} "
              f"{s['faithfulness']:<8.3f} {s['answer_relevancy']:<8.3f} "
              f"{s['context_precision']:<8.3f} {s['context_recall']:<8.3f}")