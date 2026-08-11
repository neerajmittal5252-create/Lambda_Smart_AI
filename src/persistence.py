from supabase import create_client, Client
from src.config import Config

supabase: Client = create_client(Config.SUPABASE_URL, Config.SUPABASE_KEY)

def save_message(thread_id: str, role: str, content: str):
    if not content:
        return
    supabase.table("chat_messages").insert({
        "thread_id": thread_id,
        "role": role,
        "content": content
    }).execute()

def load_thread_messages(thread_id: str) -> list[dict]:
    result = supabase.table("chat_messages") \
        .select("role,content") \
        .eq("thread_id", thread_id) \
        .order("created_at") \
        .execute()
    return result.data

def get_all_threads() -> list[str]:
    result = supabase.table("chat_messages") \
        .select("thread_id") \
        .order("created_at", desc=True) \
        .execute()
    seen = []
    for row in result.data:
        tid = row["thread_id"]
        if tid not in seen:
            seen.append(tid)
    return seen
