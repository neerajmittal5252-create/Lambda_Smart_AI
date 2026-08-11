import os
import streamlit as st
from dotenv import load_dotenv
load_dotenv()
def get_config(key: str, default: str = None) -> str:
    return os.getenv(key) or st.secrets.get(key, default) if hasattr(st, "secrets") else os.getenv(key, default)

class Config:
    SUPABASE_URL=get_config("SUPABASE_URL")
    SUPABASE_KEY=get_config("SUPABASE_KEY")
    OPENROUTER_API_KEY=get_config("OPENROUTER_API_KEY")
    OPENROUTER_BASE_URL="https://openrouter.ai/api/v1"
    EMBEDDING_MODEL="nvidia/nemotron-3-embed-1b:free"
    VECTOR_DIM=1024  
    VECTOR_TABLE="repo_embeddings"
    LANGCHAIN_TRACING_V2=get_config("LANGCHAIN_TRACING_V2", "false")
    LANGCHAIN_API_KEY=get_config("LANGCHAIN_API_KEY")
    LANGCHAIN_PROJECT=get_config("LANGCHAIN_PROJECT", "lambda-chat-ai")
    REPO_CLONE_DIR="./cloned_repos"
