from crewai import Agent, Task, Crew, LLM, Process
from crewai.tools import tool

from src.config import Config
from src.vector_store import search_similar


llm = LLM(
    model="openrouter/openai/gpt-4o-mini",
    base_url=Config.OPENROUTER_BASE_URL,
    api_key=Config.OPENROUTER_API_KEY,
    temperature=0.2
)


def build_search_repo_tool(repo_name: str):

    @tool("Search Repository")
    def search_repo_tool(query: str) -> str:
        """Search the repository's ingested code for relevant information."""

        try:
            results = search_similar(
                query,
                top_k=5,
                repo_name=repo_name
            )

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


def run_repo_crew(
    repo_name: str,
    question: str | None = None
):

    search_tool = build_search_repo_tool(repo_name)

    analyst = Agent(
        role="Code Analyst",
        goal=(
            f"Answer questions about the '{repo_name}' repository "
            "using only code retrieved from the vector database."
        ),
        backstory=(
            "You are a senior software engineer who understands "
            "software architecture, source code, dependencies, APIs, "
            "and project structure. Never guess."
        ),
        tools=[search_tool],
        llm=llm,
        verbose=True,
        allow_delegation=False
    )

    if question:

        analysis_description = (
            f"Answer the following question about the repository "
            f"'{repo_name}':\n\n"
            f"{question}\n\n"
            "Use the Search Repository tool to retrieve relevant code. "
            "Use multiple searches with different queries if necessary. "
            "Answer directly and cite the relevant file paths. "
            "If the retrieved code does not contain enough information, "
            "say so explicitly instead of guessing."
        )

        expected_output = (
            f"A direct answer to the question about '{repo_name}', "
            "based only on retrieved repository code, with relevant "
            "file paths."
        )

    else:

        analysis_description = (
            f"Analyze the repository '{repo_name}'. "
            "Use the Search Repository tool multiple times to retrieve "
            "relevant code and understand the project's architecture. "
            "Investigate the main application flow, important modules, "
            "dependencies, APIs, databases, and major components. "
            "Provide a concise architectural summary."
        )

        expected_output = (
            f"A concise architectural summary of '{repo_name}' "
            "based only on retrieved repository code."
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
    # CrewOutput object — get raw string
    return str(result.raw if hasattr(result, "raw") else result)
