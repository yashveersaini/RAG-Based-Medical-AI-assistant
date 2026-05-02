import psycopg2
from psycopg2.extras import RealDictCursor
from contextlib import contextmanager
from config import settings


# ── Connection ────────────────────────────────────────────────────

def get_connection():
    return psycopg2.connect(settings.database_url)


@contextmanager
def get_db():
    conn = get_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ── Schema setup (run once) ───────────────────────────────────────

def init_db():
    with get_db() as conn:
        cur = conn.cursor()

        cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id         SERIAL PRIMARY KEY,
                email      VARCHAR(255) UNIQUE NOT NULL,
                name       VARCHAR(255) NOT NULL,
                password   VARCHAR(255) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id         SERIAL PRIMARY KEY,
                user_id    INTEGER REFERENCES users(id) ON DELETE CASCADE,
                title      VARCHAR(255) DEFAULT 'New Chat',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id         SERIAL PRIMARY KEY,
                session_id INTEGER REFERENCES sessions(id) ON DELETE CASCADE,
                role       VARCHAR(10) NOT NULL,   -- 'user' or 'assistant'
                content    TEXT NOT NULL,
                sources    TEXT,                   -- JSON string of citations
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        print("Database tables ready.")


# ── User queries ──────────────────────────────────────────────────

def create_user(email: str, name: str, hashed_password: str) -> dict:
    with get_db() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(
            "INSERT INTO users (email, name, password) VALUES (%s, %s, %s) RETURNING *",
            (email, name, hashed_password)
        )
        return dict(cur.fetchone())


def get_user_by_email(email: str) -> dict | None:
    with get_db() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("SELECT * FROM users WHERE email = %s", (email,))
        row = cur.fetchone()
        return dict(row) if row else None


def get_user_by_id(user_id: int) -> dict | None:
    with get_db() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("SELECT id, email, name, created_at FROM users WHERE id = %s", (user_id,))
        row = cur.fetchone()
        return dict(row) if row else None


# ── Session queries ───────────────────────────────────────────────

def create_session(user_id: int, title: str = "New Chat") -> dict:
    with get_db() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(
            "INSERT INTO sessions (user_id, title) VALUES (%s, %s) RETURNING *",
            (user_id, title)
        )
        return dict(cur.fetchone())


def get_user_sessions(user_id: int) -> list:
    with get_db() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(
            "SELECT * FROM sessions WHERE user_id = %s ORDER BY created_at DESC",
            (user_id,)
        )
        return [dict(r) for r in cur.fetchall()]


def update_session_title(session_id: int, title: str):
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            "UPDATE sessions SET title = %s WHERE id = %s",
            (title, session_id)
        )


def delete_session(session_id: int, user_id: int):
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            "DELETE FROM sessions WHERE id = %s AND user_id = %s",
            (session_id, user_id)
        )


# ── Message queries ───────────────────────────────────────────────

def save_message(session_id: int, role: str, content: str, sources: str = None) -> dict:
    with get_db() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(
            "INSERT INTO messages (session_id, role, content, sources) VALUES (%s, %s, %s, %s) RETURNING *",
            (session_id, role, content, sources)
        )
        return dict(cur.fetchone())


def get_session_messages(session_id: int) -> list:
    with get_db() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(
            "SELECT * FROM messages WHERE session_id = %s ORDER BY created_at ASC",
            (session_id,)
        )
        return [dict(r) for r in cur.fetchall()]
