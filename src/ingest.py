import os
from pathlib import Path
from git import Repo
from langchain.text_splitter import RecursiveCharacterTextSplitter
from src.vector_store import embed_and_store
from src.config import Config

def clone_repo(repo_url: str, repo_name: str) -> str:
    target_dir = os.path.join(Config.REPO_CLONE_DIR, repo_name)
    if os.path.exists(target_dir):
        repo = Repo(target_dir)
        repo.remotes.origin.pull()
    else:
        os.makedirs(Config.REPO_CLONE_DIR, exist_ok=True)
        Repo.clone_from(repo_url, target_dir)
    return target_dir

def read_code_files(repo_path: str):
    extensions = {".py", ".js", ".ts", ".jsx", ".tsx", ".md", ".txt", ".json", ".yaml", ".yml"}
    files = []
    for path in Path(repo_path).rglob("*"):
        if path.is_file() and path.suffix in extensions and ".git" not in str(path):
            try:
                content = path.read_text(encoding="utf-8", errors="ignore")
                files.append({"path": str(path.relative_to(repo_path)), "content": content})
            except Exception:
                continue
    return files

def chunk_and_ingest(repo_path: str, repo_name: str):
    files = read_code_files(repo_path)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    for file in files:
        chunks = splitter.split_text(file["content"])
        for chunk in chunks:
            embed_and_store(chunk, {"repo": repo_name, "file_path": file["path"]})
    return len(files)