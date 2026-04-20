from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager
from src.rag_chain import build_rag_chain, run_query
from src.ingestion import ingest_pdf_bytes
from pathlib import Path
from src.ingestion import ingest_file_from_bytes


# ── Pydantic schemas ──────────────────────────────────────────────
class AskRequest(BaseModel):
    query: str

class Source(BaseModel):
    filename: str
    page: int
    preview: str

class AskResponse(BaseModel):
    answer: str
    sources: list[Source]

class IngestResponse(BaseModel):
    status: str
    filename: str
    chunks_indexed: int


# ── App state (replaces globals in your old app.py) ───────────────
rag_chain = None
docsearch = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # runs once at startup — replaces @app.before_first_request
    global rag_chain, docsearch
    print("Loading RAG chain...")
    rag_chain, docsearch = build_rag_chain()
    print("RAG chain ready.")
    yield
    # cleanup on shutdown (if needed)


# ── App init ──────────────────────────────────────────────────────
app = FastAPI(
    title="Medical RAG Assistant",
    description="Document Q&A with citation tracking",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")


# ── Routes ────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def index():
    with open("templates/chat.html") as f:
        return f.read()


@app.post("/ask", response_model=AskResponse)
async def ask(body: AskRequest):
    if not body.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    result = await run_query(rag_chain, body.query)
    return AskResponse(
        answer=result["answer"],
        sources=[Source(**s) for s in result["sources"]]
    )


@app.post("/ingest", response_model=IngestResponse)
async def ingest(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    ext = Path(file.filename).suffix.lower()

    if ext not in {".pdf", ".docx", ".txt"}:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: .pdf .docx .txt"
        )

    file_bytes = await file.read()
    background_tasks.add_task(
        ingest_file_from_bytes,
        file_bytes,
        file.filename
    )

    return IngestResponse(
        status="processing — searchable in ~30 seconds",
        filename=file.filename,
        chunks_indexed=0
    )

@app.get("/health")
async def health():
    return {"status": "ok", "index": "medical-assistant"}