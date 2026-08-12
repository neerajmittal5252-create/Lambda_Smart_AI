from crewai import Agent, Task, Crew, LLM, Process
from crewai.tools import tool
from pathlib import Path
import os

from src.config import Config
from src.vector_store import search_similar
from src.ingest import clone_repo, read_code_files


llm = LLM(
    model="openrouter/openai/gpt-4o-mini",
    base_url=Config.OPENROUTER_BASE_URL,
    api_key=Config.OPENROUTER_API_KEY,
    temperature=0.2
)


def _ensure_repo_cloned(repo_url: str, repo_name: str) -> str:
    """Clone or pull the repo and return the local path."""
    return clone_repo(repo_url, repo_name)


def _list_repo_files(repo_path: str) -> list[dict]:
    """List all readable code files in the repo with their paths."""
    extensions = {".py", ".js", ".ts", ".jsx", ".tsx", ".md", ".txt", ".json", ".yaml", ".yml", ".rs", ".go", ".java", ".cpp", ".c", ".h", ".cs", ".rb", ".php", ".swift", ".kt", ".scala", ".sh", ".dockerfile", ".toml", ".cfg", ".ini"}
    files = []
    for path in Path(repo_path).rglob("*"):
        if path.is_file() and path.suffix.lower() in extensions and ".git" not in str(path):
            rel = str(path.relative_to(repo_path))
            files.append({"path": rel, "full_path": str(path)})
    return files


def _read_file(repo_path: str, file_path: str, max_chars: int = 8000) -> str:
    """Read a specific file from the cloned repo."""
    target = Path(repo_path) / file_path
    # Security: prevent directory traversal
    try:
        target.resolve().relative_to(Path(repo_path).resolve())
    except ValueError:
        return "Error: Invalid file path (directory traversal attempt)."

    if not target.exists() or not target.is_file():
        return f"Error: File '{file_path}' not found in repository."

    try:
        content = target.read_text(encoding="utf-8", errors="ignore")
        if len(content) > max_chars:
            content = content[:max_chars] + f"\n\n... [truncated, total length: {len(content)} chars]"
        return content
    except Exception as e:
        return f"Error reading file: {e}"


def _get_repo_structure(repo_path: str) -> str:
    """Generate a tree-like structure of the repo."""
    lines = []
    for path in sorted(Path(repo_path).rglob("*")):
        if ".git" in str(path):
            continue
        rel = path.relative_to(repo_path)
        depth = len(rel.parts) - 1
        indent = "  " * depth
        if path.is_dir():
            lines.append(f"{indent}[DIR] {rel.name}/")
        else:
            lines.append(f"{indent}[FILE] {rel.name}")
    return "\n".join(lines)


def build_repo_tools(repo_url: str, repo_name: str):
    """Build a set of CrewAI tools for exploring a repo directly."""

    # Ensure repo is cloned once
    repo_path = _ensure_repo_cloned(repo_url, repo_name)

    @tool("Search Repository Vector Store")
    def search_vector_tool(query: str) -> str:
        """Search the pre-ingested repository vector database for relevant code chunks."""
        try:
            results = search_similar(query, top_k=5, repo_name=repo_name)
            if not results:
                return "No results found in vector store for this query."
            out = []
            for r in results:
                meta = r.get("metadata") or {}
                out.append(f"File: {meta.get('file_path', 'unknown')}\n{r.get('content', '')}")
            return "\n\n---\n\n".join(out)
        except Exception as e:
            return f"Vector search failed: {e}"

    @tool("List Repository Files")
    def list_files_tool(query: str = "") -> str:
        """List all readable code files in the repository. Use this to discover what files exist."""
        files = _list_repo_files(repo_path)
        if not files:
            return "No readable code files found in the repository."
        return "\n".join(f"- {f['path']}" for f in files)

    @tool("Read Repository File")
    def read_file_tool(file_path: str) -> str:
        """Read the full contents of a specific file in the repository. Provide the relative file path."""
        return _read_file(repo_path, file_path)

    @tool("Get Repository Structure")
    def repo_structure_tool(query: str = "") -> str:
        """Get the directory tree / project structure of the repository."""
        return _get_repo_structure(repo_path)

    return [search_vector_tool, list_files_tool, read_file_tool, repo_structure_tool]


def run_repo_crew(
    repo_url: str,
    repo_name: str,
    question: str | None = None
):
    """
    Run the Repo Crew to analyze a GitHub repository.
    
    The crew will:
    1. Clone the repo if not already present
    2. Try vector store search first (if ingested)
    3. Fall back to reading files directly from disk
    4. Explore structure, read key files, and answer the question
    """
    tools = build_repo_tools(repo_url, repo_name)

    analyst = Agent(
        role="Code Analyst",
        goal=(
            f"Thoroughly analyze the '{repo_name}' repository and answer questions "
            "by reading actual source code files when needed."
        ),
        backstory=(
            "You are a senior software engineer with deep expertise in reading and understanding "
            "source code. When a repository lacks README, description, or documentation, you "
            "explore the file structure, read key source files, and infer the architecture, "
            "purpose, APIs, dependencies, and design patterns. You never guess — you read the code."
        ),
        tools=tools,
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=15,
    )

    if question:
        analysis_description = (
            f"Answer the following question about the repository '{repo_name}':\n\n"
            f"{question}\n\n"
            "Follow this strategy:\n"
            "1. First, try the 'Search Repository Vector Store' tool for quick RAG results.\n"
            "2. If RAG returns nothing useful, use 'Get Repository Structure' to see the project layout.\n"
            "3. Use 'List Repository Files' to discover all code files.\n"
            "4. Read the most relevant files using 'Read Repository File' — start with entry points "
            "(main.py, app.py, index.js, package.json, requirements.txt, etc.) and key modules.\n"
            "5. Read enough files to form a complete, accurate answer.\n"
            "6. Cite specific file paths and line references in your answer."
        )
        expected_output = (
            f"A detailed, accurate answer to the question about '{repo_name}', "
            "based on actual source code, with cited file paths."
        )
    else:
        analysis_description = (
            f"Provide a complete architectural summary of the repository '{repo_name}'.\n\n"
            "Follow this strategy:\n"
            "1. First, try the 'Search Repository Vector Store' tool.\n"
            "2. If RAG is empty, use 'Get Repository Structure' to see the layout.\n"
            "3. Use 'List Repository Files' to discover files.\n"
            "4. Read key files: entry points, config files, main modules, and core logic.\n"
            "5. Identify: purpose, tech stack, main components, data flow, APIs, and dependencies.\n"
            "6. Provide a structured summary with cited file paths."
        )
        expected_output = (
            f"A comprehensive architectural summary of '{repo_name}' "
            "based on actual source code, with cited file paths."
        )

    analysis_task = Task(
        description=analysis_description,
        expected_output=expected_output,
        agent=analyst
    )

    crew = Crew(
        agents=[analyst],
        tasks=[analysis_task],
        process=Process.sequential,
        verbose=True,
        memory=False
    )

    result = crew.kickoff()
    return str(result.raw if hasattr(result, "raw") else result)
