import uuid
from openai import OpenAI
from supabase import create_client
from src.config import Config

Config.validate()
supabase = create_client(
    Config.SUPABASE_URL,
    Config.SUPABASE_KEY
)
embedding_client = OpenAI(
    api_key=Config.OPENROUTER_API_KEY,
    base_url=Config.OPENROUTER_BASE_URL
)


def generate_embedding(text: str):
    if not text or not text.strip():
        raise ValueError("Text cannot be empty.")

    try:
        response = embedding_client.embeddings.create(
            model=Config.EMBEDDING_MODEL,
            input=text,
            encoding_format="float"
        )
    except Exception as e:
        raise RuntimeError(
            f"Embedding request failed: {e}"
        ) from e

    if not response.data:
        raise RuntimeError(
            "Embedding API returned no data."
        )

    vector = response.data[0].embedding

    if len(vector) != Config.VECTOR_DIM:
        raise ValueError(
            f"Embedding dimension mismatch. "
            f"Expected {Config.VECTOR_DIM}, "
            f"received {len(vector)}."
        )

    return vector


def embed_and_store(text: str, metadata: dict):
    vector = generate_embedding(text)

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


def search_similar(query: str, top_k: int = 5):
    vector = generate_embedding(query)

    result = supabase.rpc(
        "match_repo_embeddings",
        {
            "query_embedding": vector,
            "match_threshold": 0.7,
            "match_count": top_k
        }
    ).execute()

    return result.data or []
