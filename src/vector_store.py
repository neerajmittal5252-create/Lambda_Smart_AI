import uuid
from supabase import create_client
from langchain_openai import OpenAIEmbeddings
from src.config import Config

supabase = create_client(Config.SUPABASE_URL, Config.SUPABASE_KEY)
embeddings = OpenAIEmbeddings(
    model=Config.EMBEDDING_MODEL,
    api_key=Config.OPENROUTER_API_KEY,
    base_url=Config.OPENROUTER_BASE_URL,
    check_embedding_ctx_length=False 
)

def embed_and_store(text: str, metadata: dict):
    vector = embeddings.embed_query(text)
    supabase.table(Config.VECTOR_TABLE).insert({
        "id": str(uuid.uuid4()),
        "content": text,
        "embedding": vector,
        "metadata": metadata
    }).execute()

def search_similar(query: str, top_k: int = 5):
    vector = embeddings.embed_query(query)
    result = supabase.rpc("match_repo_embeddings", {
        "query_embedding": vector,
        "match_threshold": 0.7,
        "match_count": top_k
    }).execute()
    return result.data
