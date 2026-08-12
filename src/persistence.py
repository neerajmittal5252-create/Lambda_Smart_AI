import json
import sqlite3
from datetime import datetime, timezone
from src.config import Config


class _PersistenceBackend:
    """Abstract persistence backend."""
    def save_message(self, thread_id: str, role: str, content: str): ...
    def load_thread_messages(self, thread_id: str) -> list[dict]: ...
    def get_all_threads(self) -> list[str]: ...


class _SupabaseBackend(_PersistenceBackend):
    def __init__(self):
        from supabase import create_client, Client
        self._client: Client = create_client(Config.SUPABASE_URL, Config.SUPABASE_KEY)

    def save_message(self, thread_id: str, role: str, content: str):
        if not content:
            return
        self._client.table("chat_messages").insert({
            "thread_id": thread_id,
            "role": role,
            "content": content
        }).execute()

    def load_thread_messages(self, thread_id: str) -> list[dict]:
        result = self._client.table("chat_messages") \
            .select("role,content") \
            .eq("thread_id", thread_id) \
            .order("created_at") \
            .execute()
        return result.data or []

    def get_all_threads(self) -> list[str]:
        result = self._client.table("chat_messages") \
            .select("thread_id") \
            .order("created_at", desc=True) \
            .execute()
        seen = []
        for row in (result.data or []):
            tid = row["thread_id"]
            if tid not in seen:
                seen.append(tid)
        return seen


class _SQLiteBackend(_PersistenceBackend):
    def __init__(self, db_path: str):
        self._db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    thread_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()

    def save_message(self, thread_id: str, role: str, content: str):
        if not content:
            return
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                "INSERT INTO chat_messages (thread_id, role, content) VALUES (?, ?, ?)",
                (thread_id, role, content)
            )
            conn.commit()

    def load_thread_messages(self, thread_id: str) -> list[dict]:
        with sqlite3.connect(self._db_path) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.execute(
                "SELECT role, content FROM chat_messages WHERE thread_id = ? ORDER BY created_at",
                (thread_id,)
            )
            return [dict(row) for row in cur.fetchall()]

    def get_all_threads(self) -> list[str]:
        with sqlite3.connect(self._db_path) as conn:
            cur = conn.execute(
                "SELECT thread_id FROM chat_messages ORDER BY created_at DESC"
            )
            seen = []
            for row in cur.fetchall():
                tid = row[0]
                if tid not in seen:
                    seen.append(tid)
            return seen


# Lazy-initialized backend
_backend: _PersistenceBackend | None = None


def _get_backend() -> _PersistenceBackend:
    global _backend
    if _backend is None:
        if Config.USE_SQLITE_FALLBACK or not Config.SUPABASE_URL or not Config.SUPABASE_KEY:
            _backend = _SQLiteBackend(Config.SQLITE_DB_PATH)
        else:
            _backend = _SupabaseBackend()
    return _backend


def save_message(thread_id: str, role: str, content: str):
    _get_backend().save_message(thread_id, role, content)


def load_thread_messages(thread_id: str) -> list[dict]:
    return _get_backend().load_thread_messages(thread_id)


def get_all_threads() -> list[str]:
    return _get_backend().get_all_threads()
