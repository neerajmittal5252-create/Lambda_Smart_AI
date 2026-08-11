import os
from dotenv import load_dotenv
load_dotenv()
class Config:
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY")
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "false")
    LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
    LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "lambda-chat-ai")

    EMBEDDING_MODEL = "text-embedding-3-small"
    VECTOR_DIM = 1536

    VECTOR_TABLE = "repo_embeddings"

    REPO_CLONE_DIR = "./cloned_repos"