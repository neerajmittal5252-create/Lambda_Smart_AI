import uuid
from openai import OpenAI
from src.config import Config

Config.validate()

_embedding_client = None
_supabase_client = None


def _get_embedding_client():
    global _embedding_client
    if _embedding_client is None:
        _embedding_client = OpenAI(
            api_key=Config.OPENROUTER_API_KEY,
            base_url=Config.OPENROUTER_BASE_URL
        )
    return _embedding_client


def _get_supabase_client():
    global _supabase_client
    if _supabase_client is None:
        from supabase import create_client
        _supabase_client = create_client(Config.SUPABASE_URL, Config.SUPABASE_KEY)
    return _supabase_client


def generate_embedding(text: str):
    if not text or not text.strip():
        raise ValueError("Text cannot be empty.")

    client = _get_embedding_client()
    try:
        response = client.embeddings.create(
            model=Config.EMBEDDING_MODEL,
            input=text,
            encoding_format="float"
        )
    except Exception as e:
        raise RuntimeError(f"Embedding request failed: {e}") from e

    if not response.data:
        raise RuntimeError("Embedding API returned no data.")

    vector = response.data[0].embedding

    if len(vector) != Config.VECTOR_DIM:
        raise ValueError(
            f"Embedding dimension mismatch. Expected {Config.VECTOR_DIM}, received {len(vector)}."
        )

    return vector


def embed_and_store(text: str, metadata: dict):
    vector = generate_embedding(text)
    supabase = _get_supabase_client()

    response = (
        supabase
        .table(Config.VECTOR_TABLE)
        .insert({
            "id": str(uuid.uuid4()),
            "content": text,
            "embedding": vector,
            "metadata": metadata
        })
        .execute()
    )

    return response.data


def search_similar(query: str, top_k: int = 5, repo_name: str | None = None):
    """
    Search vector store. If repo_name is provided, filter results to that repo.
    """
    vector = generate_embedding(query)
    supabase = _get_supabase_client()

    result = supabase.rpc(
        "match_repo_embeddings",
        {
            "query_embedding": vector,
            "match_threshold": 0.7,
            "match_count": top_k
        }
    ).execute()

    data = result.data or []

    # Client-side filter by repo_name if provided
    if repo_name:
        filtered = []
        for row in data:
            meta = row.get("metadata") or {}
            if meta.get("repo") == repo_name:
                filtered.append(row)
        data = filtered

    return data
