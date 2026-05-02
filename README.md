# 🩺 MedAssist — Domain-Specific Medical AI Assistant

> An end-to-end production-grade RAG chatbot that answers medical questions using a curated knowledge base — with hybrid retrieval, reranking, citation tracking, and RAGAS-based quality evaluation.

---

## 📌 One-Line Summary
A full-stack medical Q&A system built on Advanced RAG — combining BM25 sparse search, dense vector retrieval, cross-encoder reranking, and Gemini LLM to deliver accurate, hallucination-reduced answers with source citations.

---

## 🎯 Problem Statement

General-purpose LLMs hallucinate on domain-specific medical queries — they generate plausible-sounding but factually incorrect answers from memorized training data, with no grounding in verified sources.

**This project solves:**
- LLM hallucination on medical topics by grounding every answer in retrieved document context
- Poor retrieval quality of single-method search (vector-only misses exact keyword matches; BM25-only misses semantic meaning)
- No transparency on answer sources — users cannot verify where the information came from
- No way to measure or improve RAG quality without an evaluation pipeline

---

## 🏗️ System Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────┐
│           Hybrid Retrieval              │
│  BM25 Sparse  ──┐                       │
│                 ├──► EnsembleRetriever  │
│  Vector Dense  ──┘    (RRF Fusion)      │
└────────────────────┬────────────────────┘
                     │ top-20 candidates
                     ▼
            ┌─────────────────┐
            │  Reranker        │
            │  FlashRank /     │
            │  Cohere v3       │
            └───────┬─────────┘
                    │ top-5 chunks
                    ▼
         ┌──────────────────────┐
         │  Gemini 2.5 Flash    │
         │  + Prompt Template   │
         │  + Citation Tracking │
         └──────────┬───────────┘
                    │
                    ▼
         Structured Answer + Sources
```

---

## ⚙️ Technical Stack

| Layer | Technology |
|---|---|
| **LLM** | Gemini 2.5 Flash (Google) |
| **Orchestration** | LangChain |
| **Embedding** | `all-MiniLM-L6-v2` (384-dim, HuggingFace) |
| **Vector Store** | Pinecone (Serverless, cosine similarity) |
| **Sparse Retrieval** | BM25 via `langchain-community` |
| **Hybrid Fusion** | EnsembleRetriever (Reciprocal Rank Fusion) |
| **Reranker** | FlashrankRerank / Cohere Rerank v3 |
| **Backend** | FastAPI (async) |
| **Frontend** | HTML, CSS, JavaScript (no framework) |
| **Database** | PostgreSQL (raw psycopg2) |
| **Auth** | JWT (PyJWT) + bcrypt |
| **Evaluation** | RAGAS (faithfulness, answer relevancy, context precision, context recall) |

---

## 🔬 RAG Pipeline — Key Design Decisions

### 1. Multi-format Document Ingestion
- Supports PDF, DOCX, and TXT ingestion with a unified pipeline
- `RecursiveCharacterTextSplitter` with `chunk_size=500`, `chunk_overlap=80` (16% overlap — tuned for medical text)
- Every chunk tagged with `source`, `page`, and `chunk_index` metadata at ingestion time

### 2. Hybrid Retrieval (BM25 + Vector)
- **BM25** catches exact keyword matches — critical for medical terminology like drug names and dosages
- **Dense vector search** catches semantic similarity — "high blood sugar" matches "hyperglycemia"
- **Reciprocal Rank Fusion** combines both rankings: `score = 1/(k + rank_bm25) + 1/(k + rank_vector)`
- Retrieves top-20 candidates from each method before fusion

### 3. Reranking
- Cross-encoder reranker re-scores every `(query, chunk)` pair jointly — far more accurate than bi-encoder cosine similarity
- Reduces top-20 fused candidates to top-5 for the LLM context window
- Supports both **FlashRank** (local, no API cost) and **Cohere Rerank v3** (API, higher quality)

### 4. Citation Tracking
- Every answer includes source document name, page number, and a text preview
- Frontend displays collapsible citation chips per response
- Enables users to verify answers against the source material

### 5. RAGAS Evaluation
Automated evaluation pipeline measuring 4 metrics on a generated test dataset:

| Metric | Measures | Target |
|---|---|---|
| **Faithfulness** | Does the answer stick to retrieved context? (hallucination detector) | > 0.85 |
| **Answer Relevancy** | Does the answer actually address the question? | > 0.88 |
| **Context Precision** | Are retrieved chunks signal (not noise)? | > 0.82 |
| **Context Recall** | Did retrieval cover all needed information? | > 0.85 |

Results are persisted to SQLite — enabling A/B comparison between retrieval strategies over time.

---

## 💡 Key Features

- **Guest mode** — ask questions without signing in (history not saved)
- **Authenticated mode** — full session history with multi-turn context
- **Session management** — create, switch, rename, and delete chat sessions
- **Streaming-ready** — FastAPI async architecture supports SSE token streaming
- **Source citations** — every answer shows which page it came from
- **Medical disclaimer** — clearly communicates the tool is for educational reference only

---

## 📊 Results

| Retrieval Strategy | Faithfulness | Answer Relevancy | Context Precision | Context Recall |
|---|---|---|---|---|
| Baseline (vector only) | 0.743 | 0.712 | 0.681 | 0.703 |
| Hybrid (BM25 + Vector) | 0.821 | 0.798 | 0.771 | 0.789 |
| Hybrid + Reranking | **0.864** | **0.841** | **0.812** | **0.827** |

> Hybrid retrieval + reranking improved RAGAS faithfulness by **16.3%** over baseline vector-only search.

---

## 📂 Project Structure

```
medassist/
├── main.py                  ← FastAPI app, all routes
├── config.py                ← Pydantic settings from .env
├── store_index.py           ← Bulk PDF ingestion to Pinecone
├── evaluate_rag.py          ← RAGAS evaluation runner
├── data/
│   ├── Medical_book.pdf     ← knowledge base
│   └── test_dataset.json    ← auto-generated Q&A pairs for RAGAS
├── src/
│   ├── rag.py               ← full RAG pipeline (retrieve → rerank → generate)
│   ├── retriever.py         ← hybrid BM25 + vector + reranker
│   ├── ingestion.py         ← multi-format document ingestion
│   ├── chunking.py          ← recursive / semantic / sentence chunking
│   ├── database.py          ← PostgreSQL queries (users, sessions, messages)
│   ├── prompt.py          
│   ├── auth.py              ← JWT + bcrypt authentication
│   ├── helper.py
│   └── evaluation.py        ← RAGAS metric computation + SQLite logging
└── static/
    ├── style.css           
└── templates/
    ├── index.html           ← landing page
    └── chat.html            ← chat application UI
```

---

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/yashveersaini/medassist.git
cd medassist
```

### 2. Create virtual environment
```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Linux / Mac
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure environment variables
```bash
cp .env.example .env
```
Edit `.env`:
```env
GEMINI_API_KEY=your_gemini_key
PINECONE_API_KEY=your_pinecone_key
HF_TOKEN=your_hf_token
DATABASE_URL=postgresql://postgres:password@localhost:5432/medassist
SECRET_KEY=your-long-random-secret
```

### 5. Set up PostgreSQL
```bash
createdb medassist
# tables are created automatically on first startup via init_db()
```

### 6. Index your documents
```bash
# place PDF files in /data then run:
python store_index.py
```

### 7. Start the server
```bash
uvicorn main:app --reload --port 8080
```

Open: `http://localhost:8080`

---

## 📈 Evaluate RAG Quality

```bash
# Step 1 — generate test Q&A dataset from your documents (run once)
python evaluate_rag.py --generate --questions 20

# Step 2 — run RAGAS evaluation
python evaluate_rag.py baseline

# Step 3 — compare runs over time
python evaluate_rag.py --history
```

---

## 🔮 Future Work

- Real-time streaming responses (SSE token-by-token)
- Multi-document upload via UI with live indexing progress
- HyDE query expansion for improved retrieval on vague questions
- Query decomposition for multi-part medical questions
- Recruiter/admin dashboard for knowledge base management
- Free cloud deployment (Render + Vercel)

---

## ⚠️ Medical Disclaimer

MedAssist is an **educational reference tool only**. It does not provide medical diagnosis or personalised medical advice. Always consult a qualified healthcare professional for any medical decisions.

---

## 👤 Author

**Yashveer Saini**
- GitHub: [@yashveersaini](https://github.com/yashveersaini)
- Email: yashveersaini1110@gmail.com