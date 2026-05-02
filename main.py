from fastapi import FastAPI, Request, Depends, HTTPException, Body
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from langchain_huggingface import HuggingFaceEmbeddings

from src.database import *
from src.auth import *
from src.rag import get_answer

embeddings = HuggingFaceEmbeddings(model_name=settings.embed_model)

app = FastAPI()
templates = Jinja2Templates(directory="templates")

@app.on_event("startup")
def startup_event():
    from src.database import init_db
    init_db()


# FRONTEND ROUTES
@app.get("/")
def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"request": request}   
    )


@app.get("/chat", response_class=HTMLResponse)
def chat_page(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="chat.html",
        context={"request": request}   
    )

# AUTH
class SignupRequest(BaseModel):
    email: str
    name: str
    password: str

class LoginRequest(BaseModel):
    email: str
    password: str

@app.post("/api/signup")
def signup(data: SignupRequest):
    existing = get_user_by_email(data.email)
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")

    hashed = hash_password(data.password)
    user = create_user(data.email, data.name, hashed)

    token = create_token(user["id"])

    response = JSONResponse({"message": "User created"})
    response.set_cookie(key="token", value=token, httponly=True)

    return response


@app.post("/api/login")
def login(data: LoginRequest):
    user = get_user_by_email(data.email)

    if not user or not verify_password(data.password, user["password"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = create_token(user["id"])

    response = JSONResponse({"message": "Login successful"})
    response.set_cookie(key="token", value=token, httponly=True)

    return response


@app.post("/api/logout")
def logout():
    response = JSONResponse({"message": "Logged out"})
    response.delete_cookie("token")
    return response


@app.get("/api/me")
def get_me(user_id: int = Depends(get_current_user)):
    if not user_id:
        return {"user": None}

    user = get_user_by_id(user_id)
    return {"user": user}


# CHAT SYSTEM

@app.post("/api/sessions")
def create_new_session(user_id: int = Depends(require_user)):
    return create_session(user_id)


@app.get("/api/sessions")
def list_sessions(user_id: int = Depends(require_user)):
    sessions = get_user_sessions(user_id)
    return {"sessions": sessions}   # ✅ frontend expects {sessions: []}


@app.delete("/api/sessions/{session_id}")
def delete_user_session(session_id: int, user_id: int = Depends(require_user)):
    delete_session(session_id, user_id)
    return {"message": "Session deleted"}


@app.get("/api/sessions/{session_id}/messages")
def get_messages(session_id: int, user_id: int = Depends(require_user)):
    messages = get_session_messages(session_id)
    return {"messages": messages}   # ✅ frontend expects {messages: []}


     
# MAIN CHAT API
     
@app.post("/api/chat")
def ask(
    data: dict = Body(...),
    user_id: int = Depends(get_current_user)
):
    try:

        query = data.get("query")
        session_id = data.get("session_id")

        if not query:
            raise HTTPException(status_code=400, detail="Query missing")


        result = get_answer(query, embeddings)

        answer = result.get("answer", "")
        sources = result.get("sources", [])

        return {
            "answer": answer,
            "sources": sources,
            "session_id": session_id,
            "logged_in": bool(user_id)
        }

    except Exception as e:
        print("🔥 FULL ERROR:", str(e))
        import traceback
        traceback.print_exc()

        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )