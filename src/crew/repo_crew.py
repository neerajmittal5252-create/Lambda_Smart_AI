from crewai import Agent, Task, Crew, Process
from crewai.tools import tool
from langchain_openai import ChatOpenAI
from src.config import Config
from src.ingest import clone_repo, chunk_and_ingest
from src.vector_store import search_similar

llm = ChatOpenAI(
    model="nvidia/nemotron-3-nano-30b-a3b:free",
    base_url="https://openrouter.ai/api/v1",
    api_key=Config.OPENROUTER_API_KEY
)

@tool("Ingest Repository")
def ingest_repo_tool(repo_url: str, repo_name: str) -> str:
    """Clone a GitHub repository and embed its code into the vector database."""
    path = clone_repo(repo_url, repo_name)
    file_count = chunk_and_ingest(path, repo_name)
    return f"Ingested {file_count} files from {repo_name} into the vector store."

@tool("Search Repository")
def search_repo_tool(query: str) -> str:
    """Search the vector store for relevant code snippets."""
    results = search_similar(query, top_k=5)
    if not results:
        return "No relevant code found."
    return "\n\n".join([
        f"File: {r['metadata'].get('file_path', 'unknown')}\n{r['content']}"
        for r in results
    ])

ingestor = Agent(
    role="Repository Ingestor",
    goal="Clone and ingest GitHub repositories into the vector database for RAG",
    backstory="You are responsible for fetching codebases and preparing them for semantic search.",
    tools=[ingest_repo_tool],
    llm=llm,
    verbose=True
)

summarizer = Agent(
    role="Code Summarizer",
    goal="Analyze ingested code and summarize the architecture and key components",
    backstory="You are a staff engineer who quickly grasps project structure from source code.",
    tools=[search_repo_tool],
    llm=llm,
    verbose=True
)

qa_agent = Agent(
    role="Repository QA",
    goal="Answer developer questions using retrieved code context from the vector store",
    backstory="You help developers by finding exact code snippets and explaining them clearly.",
    tools=[search_repo_tool],
    llm=llm,
    verbose=True
)

def run_repo_crew(repo_url: str, repo_name: str):
    """Run ingestion + summarization crew."""
    ingest_task = Task(
        description=f"Clone and ingest the repo {repo_url} with name {repo_name}.",
        expected_output="Confirmation string with number of files ingested.",
        agent=ingestor
    )

    summary_task = Task(
        description=f"Search the vector DB for {repo_name} and summarize what this project does.",
        expected_output="A concise 3-5 sentence architectural summary.",
        agent=summarizer,
        context=[ingest_task]
    )

    crew = Crew(
        agents=[ingestor, summarizer],
        tasks=[ingest_task, summary_task],
        process=Process.sequential,
        verbose=True
    )

    return crew.kickoff(inputs={"repo_url": repo_url, "repo_name": repo_name})