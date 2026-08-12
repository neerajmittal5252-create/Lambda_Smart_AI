from crewai import Agent, Task, Crew, LLM, Process
from crewai.tools import tool

from src.config import Config
from src.ingest import clone_repo, chunk_and_ingest
from src.vector_store import search_similar


llm = LLM(
    model="openrouter/openai/gpt-4o-mini",
    base_url=Config.OPENROUTER_BASE_URL,
    api_key=Config.OPENROUTER_API_KEY,
    temperature=0.2
)


@tool("Ingest Repository")
def ingest_repo_tool(repo_url: str, repo_name: str) -> str:
    """Clone a GitHub repository and ingest its code into the vector database."""

    try:
        path = clone_repo(repo_url, repo_name)
        file_count = chunk_and_ingest(path, repo_name)

        return (
            f"Repository '{repo_name}' was successfully ingested. "
            f"Files processed: {file_count}."
        )

    except Exception as e:
        return f"Repository ingestion failed: {str(e)}"


@tool("Search Repository")
def search_repo_tool(query: str) -> str:
    """Search the vector database for relevant repository code."""

    try:
        results = search_similar(query, top_k=5)

        if not results:
            return "No relevant code found."

        output = []

        for result in results:
            metadata = result.get("metadata") or {}
            file_path = metadata.get("file_path", "unknown")
            content = result.get("content", "")

            output.append(
                f"File: {file_path}\n{content}"
            )

        return "\n\n".join(output)

    except Exception as e:
        return f"Repository search failed: {str(e)}"


ingestor = Agent(
    role="Repository Ingestor",
    goal="Clone GitHub repositories and ingest their source code into the vector database.",
    backstory=(
        "You are responsible for fetching repositories, processing their source "
        "files, and preparing them for semantic search."
    ),
    tools=[ingest_repo_tool],
    llm=llm,
    verbose=True,
    allow_delegation=False
)


summarizer = Agent(
    role="Code Summarizer",
    goal="Analyze repository code retrieved from the vector database and summarize its architecture and important components.",
    backstory=(
        "You are a senior software engineer who understands software architecture, "
        "source code, dependencies, APIs, and project structure."
    ),
    tools=[search_repo_tool],
    llm=llm,
    verbose=True,
    allow_delegation=False
)


def run_repo_crew(repo_url: str, repo_name: str):

    ingest_task = Task(
        description=(
            f"Clone the GitHub repository at {repo_url} using the repository name "
            f"{repo_name}. Ingest its source code into the vector database. "
            f"Use the Ingest Repository tool. Do not attempt to summarize the "
            f"repository before ingestion is complete."
        ),
        expected_output=(
            "A confirmation that the repository was successfully cloned and "
            "ingested, including the number of processed files."
        ),
        agent=ingestor
    )

    summary_task = Task(
        description=(
            f"Analyze the repository '{repo_name}' after it has been ingested. "
            f"Use the Search Repository tool multiple times to retrieve relevant "
            f"code and understand the project's architecture. Investigate the "
            f"main application flow, important modules, dependencies, database "
            f"or API integrations, and major components. Then provide a concise "
            f"3-5 sentence architectural summary based only on the retrieved code."
        ),
        expected_output=(
            "A concise 3-5 sentence architectural summary describing the "
            "repository's purpose, architecture, major components, and important "
            "technologies."
        ),
        agent=summarizer,
        context=[ingest_task]
    )

    crew = Crew(
        agents=[ingestor, summarizer],
        tasks=[ingest_task, summary_task],
        process=Process.sequential,
        verbose=True,
        memory=False
    )

    return crew.kickoff()
