import jwt
import bcrypt
from datetime import datetime, timedelta
from fastapi import HTTPException, Cookie
from typing import Optional
from config import settings


SECRET_KEY = settings.secret_key
ALGORITHM = "HS256"
TOKEN_EXPIRE_DAYS = 7


# ── Password ──────────────────────────────────────────────────────

def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode(), hashed.encode())


# ── JWT token ─────────────────────────────────────────────────────

def create_token(user_id: int) -> str:
    payload = {
        "user_id": user_id,
        "exp": datetime.utcnow() + timedelta(days=TOKEN_EXPIRE_DAYS)
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> int:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload["user_id"]
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Session expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")


# ── FastAPI dependency ────────────────────────────────────────────

def get_current_user(token: Optional[str] = Cookie(default=None)) -> Optional[int]:
    """Returns user_id if logged in, None if guest."""
    if not token:
        return None
    try:
        return decode_token(token)
    except HTTPException:
        return None


def require_user(token: Optional[str] = Cookie(default=None)) -> int:
    """Raises 401 if not logged in."""
    if not token:
        raise HTTPException(status_code=401, detail="Please sign in")
    return decode_token(token)
