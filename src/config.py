import os
from dotenv import load_dotenv
load_dotenv()


def get_config(key: str, default=None):
    """Get configuration from environment variables or default value."""
    return os.getenv(key, default)


class Config:
    SUPABASE_URL = get_config("SUPABASE_URL")
    SUPABASE_KEY = get_config("SUPABASE_KEY")
    OPENROUTER_API_KEY = get_config("OPENROUTER_API_KEY")
    OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
    EMBEDDING_MODEL = "nvidia/nemotron-3-embed-1b:free"
    VECTOR_DIM = 2048
    VECTOR_TABLE = "repo_embeddings"
    LANGCHAIN_TRACING_V2 = get_config("LANGCHAIN_TRACING_V2", "false")
    LANGCHAIN_API_KEY = get_config("LANGCHAIN_API_KEY")
    LANGCHAIN_PROJECT = get_config("LANGCHAIN_PROJECT", "lambda-chat-ai")
    REPO_CLONE_DIR = "./cloned_repos"
    USE_SQLITE_FALLBACK = get_config("USE_SQLITE_FALLBACK", "false").lower() == "true"
    SQLITE_DB_PATH = get_config("SQLITE_DB_PATH", "./lambda_chat.db")

    @classmethod
    def validate(cls):
        """Validate required configuration."""
        required = {
            "OPENROUTER_API_KEY": cls.OPENROUTER_API_KEY,
        }
        missing = [key for key, value in required.items() if not value]
        if missing:
            raise ValueError(
                "Missing required configuration: " + ", ".join(missing)
            )
