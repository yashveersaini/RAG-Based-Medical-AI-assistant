import json
import os
from datetime import datetime
from pathlib import Path
from typing import List
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from config import settings
from src.prompt import qa_generation_prompt
import time  
import random
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()


# ── 1. Generate Test Dataset ──────────────────────────────────────

def generate_test_dataset(chunks: List[Document], num_questions: int = 100) -> List[dict]:
    """
    Generates Q&A pairs and saves to JSON after EVERY successful generation.
    """
    # llm = ChatGoogleGenerativeAI(
    #     model="gemini-2.5-flash",
    #     google_api_key=settings.gemini_api_key
    # )

    llm = HuggingFaceEndpoint(
        repo_id="MiniMaxAI/MiniMax-M2.5",
        task="text-generation"
    )

    model = ChatHuggingFace(llm=llm)

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
            response = model.invoke(prompt)
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
    """Runs the actual RAG pipeline to get answers for evaluation."""
    results = []
    
    for i, item in enumerate(test_data):
        print(f"Processing question {i+1}/{len(test_data)}")
        
        # 1. Retrieve
        docs = retriever.invoke(item["question"])
        contexts = [doc.page_content for doc in docs]
        
        # 2. Format context for the chain (matches your main script)
        formatted_context = "\n\n".join([f"Doc {j+1}: {c}" for j, c in enumerate(contexts)])
        
        # 3. Generate Answer
        response = chain.invoke({"context": formatted_context, "query": item["question"]})
        
        results.append({
            "question": item["question"],
            "answer": response.content,      # What LLM generated
            "contexts": contexts,            # What retriever found
            "ground_truth": item["ground_truth"] # What was expected
        })
        
    return results

# ── 3. Score with RAGAS ───────────────────────────────────────────

def evaluate_ragas(results: List[dict]) -> dict:
    """Scores the results using RAGAS metrics."""
    dataset = Dataset.from_list(results)
    
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    embeddings = HuggingFaceEmbeddings(model_name=settings.embed_model)

    print("Calculating RAGAS scores...")
    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=llm,
        embeddings=embeddings,
    )
    
    # Clean output dictionary
    scores = {k: round(float(v), 4) for k, v in result.items()}
    return scores

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