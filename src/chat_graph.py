import os
import re
from typing import TypedDict, Annotated, Literal, Optional

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from src.config import Config
from src.vector_store import search_similar
from src.persistence import save_message, load_thread_messages
from src.crew.repo_crew import run_repo_crew


if Config.LANGCHAIN_TRACING_V2.lower() == "true":
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = Config.LANGCHAIN_API_KEY or ""
    os.environ["LANGCHAIN_PROJECT"] = Config.LANGCHAIN_PROJECT


model = ChatOpenAI(
    model="nvidia/nemotron-3-nano-30b-a3b:free",
    base_url="https://openrouter.ai/api/v1",
    api_key=Config.OPENROUTER_API_KEY,
)

router_model = ChatOpenAI(
    model="nvidia/nemotron-3-nano-30b-a3b:free",
    base_url="https://openrouter.ai/api/v1",
    api_key=Config.OPENROUTER_API_KEY,
    temperature=0,
)


search_tool = DuckDuckGoSearchRun(region="us-en")


@tool
def search_repo_knowledge(query: str) -> str:
    """Search ingested repository code/docs for content relevant to the query."""
    results = search_similar(query, top_k=5)

    if not results:
        return "No relevant repository context found."

    return "\n---\n".join(
        f"File: {r['metadata'].get('file_path', 'unknown')}\n{r['content']}"
        for r in results
    )


llm_tools = [search_tool, search_repo_knowledge]
model_with_tools = model.bind_tools(llm_tools)


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    thread_id: str
    repo_url: Optional[str]
    repo_name: Optional[str]
    intent: Optional[Literal["general", "repo"]]
    final_response: Optional[str]


def input_node(state: ChatState) -> dict:
    return {}


def memory_node(state: ChatState) -> dict:
    thread_id = state["thread_id"]
    history = load_thread_messages(thread_id)

    if len(state["messages"]) > 1:
        return {}

    restored: list[BaseMessage] = []

    for row in history:
        if row["role"] == "user":
            restored.append(HumanMessage(content=row["content"]))
        elif row["role"] == "assistant":
            restored.append(AIMessage(content=row["content"]))

    latest = state["messages"][-1]

    save_message(
        thread_id,
        "user",
        latest.content
    )

    return {
        "messages": restored + [latest]
    }


ROUTER_PROMPT = (
    "Classify the user's latest message into exactly one label:\n"
    "- 'repo': the user is asking to summarize, explain, ingest, or answer "
    "questions about a GitHub repository/codebase.\n"
    "- 'general': anything else (normal chat, general questions, web lookups).\n"
    "Respond with only the single word 'repo' or 'general'."
)


GITHUB_URL_RE = re.compile(
    r"https?://github\.com/([\w.-]+)/([\w.-]+?)(?:\.git)?(?:[/\s]|$)"
)


def router_node(state: ChatState) -> dict:
    last_user_msg = next(
        (
            m
            for m in reversed(state["messages"])
            if isinstance(m, HumanMessage)
        ),
        None,
    )

    if last_user_msg is None:
        return {"intent": "general"}

    updates: dict = {}

    match = GITHUB_URL_RE.search(last_user_msg.content)

    if match and not (
        state.get("repo_url")
        and state.get("repo_name")
    ):
        owner, name = match.group(1), match.group(2)

        updates["repo_url"] = f"https://github.com/{owner}/{name}"
        updates["repo_name"] = name

    has_repo_context = bool(
        updates.get("repo_url")
        or (
            state.get("repo_url")
            and state.get("repo_name")
        )
    )

    if not has_repo_context:
        updates["intent"] = "general"
        return updates

    try:
        result = router_model.invoke(
            [
                SystemMessage(content=ROUTER_PROMPT),
                last_user_msg,
            ]
        )

        label = result.content.strip().lower()

        updates["intent"] = (
            "repo"
            if "repo" in label
            else "general"
        )

    except Exception as e:
        updates["intent"] = (
            "repo"
            if match
            else "general"
        )

    return updates


def route_after_classification(state: ChatState) -> str:
    return (
        "tool_node"
        if state.get("intent") == "repo"
        else "llm_node"
    )


def llm_node(state: ChatState) -> dict:
    try:
        response = model_with_tools.invoke(
            state["messages"]
        )

        if getattr(response, "tool_calls", None):
            tool_messages = []

            for call in response.tool_calls:
                fn = {
                    t.name: t
                    for t in llm_tools
                }[call["name"]]

                result = fn.invoke(call["args"])

                tool_messages.append({
                    "role": "tool",
                    "content": str(result),
                    "tool_call_id": call["id"],
                })

            follow_up = model_with_tools.invoke(
                state["messages"]
                + [response]
                + tool_messages
            )

            return {
                "messages": [
                    response,
                    *tool_messages,
                    follow_up,
                ]
            }

        return {
            "messages": [response]
        }

    except Exception as e:
        return {
            "messages": [
                AIMessage(
                    content=_friendly_error(e)
                )
            ]
        }


def tool_node(state: ChatState) -> dict:
    last_user_msg = next(
        (
            m
            for m in reversed(state["messages"])
            if isinstance(m, HumanMessage)
        ),
        None,
    )

    question = (
        last_user_msg.content
        if last_user_msg
        else None
    )

    try:
        result = run_repo_crew(
            repo_url=state["repo_url"],
            repo_name=state["repo_name"],
            question=question,
        )

        return {
            "messages": [
                AIMessage(content=str(result))
            ]
        }

    except Exception as e:
        return {
            "messages": [
                AIMessage(
                    content=_friendly_error(e)
                )
            ]
        }


def _friendly_error(e: Exception) -> str:
    msg = str(e)

    if (
        "429" in msg
        or "RateLimitError" in type(e).__name__
        or "rate limit" in msg.lower()
    ):
        return (
            "⚠️ The free OpenRouter model has hit its daily request limit. "
            "Please wait for the daily reset, or add credits to your OpenRouter "
            "account to raise the limit, then try again."
        )

    return (
        f"⚠️ Something went wrong while generating a response: {msg}"
    )


def response_node(state: ChatState) -> dict:
    final_msg = state["messages"][-1]

    save_message(
        state["thread_id"],
        "assistant",
        final_msg.content
    )

    return {
        "final_response": final_msg.content
    }


graph = StateGraph(ChatState)

graph.add_node("input_node", input_node)
graph.add_node("memory_node", memory_node)
graph.add_node("router_node", router_node)
graph.add_node("llm_node", llm_node)
graph.add_node("tool_node", tool_node)
graph.add_node("response_node", response_node)

graph.add_edge(START, "input_node")
graph.add_edge("input_node", "memory_node")
graph.add_edge("memory_node", "router_node")

graph.add_conditional_edges(
    "router_node",
    route_after_classification,
    {
        "llm_node": "llm_node",
        "tool_node": "tool_node",
    },
)

graph.add_edge("llm_node", "response_node")
graph.add_edge("tool_node", "response_node")
graph.add_edge("response_node", END)

chatbot = graph.compile()
