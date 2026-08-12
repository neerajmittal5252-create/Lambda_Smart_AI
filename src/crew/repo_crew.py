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


def build_search_repo_tool(repo_name: str):
    """Build a search tool bound to a single repo.

    Binding repo_name via closure (rather than trusting the LLM to pass it
    correctly on every call) is what makes retrieval repo-scoped. Without
    this, search_similar() queries the vector store globally and can return
    chunks from a different ingested repo, producing answers/summaries about
    the wrong codebase.
    """

    @tool("Search Repository")
    def search_repo_tool(query: str) -> str:
        f"""Search the '{repo_name}' repository's ingested code for content relevant to `query`."""

        try:
            results = search_similar(query, top_k=5, repo_name=repo_name)

            if not results:
                return f"No relevant code found in '{repo_name}' for that query."

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

    return search_repo_tool


def build_agents(repo_name: str):
    search_tool = build_search_repo_tool(repo_name)

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

    analyst = Agent(
        role="Code Analyst",
        goal=(
            f"Answer questions about the '{repo_name}' repository using only code "
            f"retrieved from the vector database, and clearly say when the "
            f"retrieved code doesn't contain the answer."
        ),
        backstory=(
            "You are a senior software engineer who understands software architecture, "
            "source code, dependencies, APIs, and project structure. You never guess "
            "or rely on general knowledge about similarly named projects -- you only "
            "answer from code you actually retrieved via the search tool."
        ),
        tools=[search_tool],
        llm=llm,
        verbose=True,
        allow_delegation=False
    )

    return ingestor, analyst


def run_repo_crew(repo_url: str, repo_name: str, question: str | None = None):
    """Ingest a repo, then either answer `question` about it or, if no
    question is given, produce a general architecture summary.

    `question` is threaded all the way into the task description and expected
    output so the analyst agent actually addresses what was asked instead of
    always returning a generic summary regardless of intent.
    """

    ingestor, analyst = build_agents(repo_name)

    ingest_task = Task(
        description=(
            f"Clone the GitHub repository at {repo_url} using the repository name "
            f"'{repo_name}'. Ingest its source code into the vector database. "
            f"Use the Ingest Repository tool. Do not attempt to analyze the "
            f"repository before ingestion is complete."
        ),
        expected_output=(
            "A confirmation that the repository was successfully cloned and "
            "ingested, including the number of processed files."
        ),
        agent=ingestor
    )

    if question:
        analysis_description = (
            f"The repository '{repo_name}' has just been ingested. Using the "
            f"Search Repository tool, search for code relevant to the following "
            f"question and issue multiple searches with different phrasings if "
            f"the first search doesn't return enough context. "
            f"Question: {question}\n\n"
            f"Answer the question directly and specifically, citing the file "
            f"paths you used. If the retrieved code does not contain enough "
            f"information to answer confidently, say so explicitly instead of "
            f"guessing or falling back to a generic project description."
        )
        analysis_expected_output = (
            f"A direct, specific answer to the question \"{question}\", grounded "
            f"only in code retrieved from '{repo_name}', citing the relevant file "
            f"paths. If the answer can't be determined from retrieved code, a "
            f"clear statement of that fact instead of a guess."
        )
    else:
        analysis_description = (
            f"Analyze the repository '{repo_name}' after it has been ingested. "
            f"Use the Search Repository tool multiple times with varied queries "
            f"to retrieve relevant code and understand the project's architecture. "
            f"Investigate the main application flow, important modules, "
            f"dependencies, database or API integrations, and major components. "
            f"Then provide a concise 3-5 sentence architectural summary based "
            f"only on the retrieved code from '{repo_name}'."
        )
        analysis_expected_output = (
            f"A concise 3-5 sentence architectural summary describing "
            f"'{repo_name}'s purpose, architecture, major components, and "
            f"important technologies, based only on retrieved code."
        )

    analysis_task = Task(
        description=analysis_description,
        expected_output=analysis_expected_output,
        agent=analyst,
        context=[ingest_task]
    )

    crew = Crew(
        agents=[ingestor, analyst],
        tasks=[ingest_task, analysis_task],
        process=Process.sequential,
        verbose=True,
        memory=False
    )

    return crew.kickoff()
