import os
from typing import TypedDict, Optional, Literal, Annotated

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
)
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from src.config import Config
from src.persistence import save_message, load_thread_messages
from src.vector_store import search_similar
from src.crew.repo_crew import run_repo_crew


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
    """Search the repository code and documentation."""

    results = search_similar(query, top_k=5)

    if not results:
        return "No relevant repository context found."

    return "\n---\n".join(
        f"File: {r['metadata'].get('file_path', 'unknown')}\n"
        f"{r['content']}"
        for r in results
    )


tools = [search_tool, search_repo_knowledge]
model_with_tools = model.bind_tools(tools)


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    thread_id: str
    repo_url: Optional[str]
    repo_name: Optional[str]
    intent: Optional[Literal["general", "repo"]]
    final_response: Optional[str]


def input_node(state: ChatState):
    return {}


def memory_node(state: ChatState):
    thread_id = state["thread_id"]
    current_message = state["messages"][-1]

    history = load_thread_messages(thread_id)

    messages = []

    for item in history:
        if item["role"] == "user":
            messages.append(
                HumanMessage(content=item["content"])
            )
        else:
            messages.append(
                AIMessage(content=item["content"])
            )

    save_message(
        thread_id,
        "user",
        current_message.content
    )

    return {
        "messages": messages + [current_message]
    }


ROUTER_PROMPT = """
Classify the user's latest message.

Return ONLY one word:

repo
- User wants to summarize, explain, analyze,
  or ask questions about a GitHub repository.

general
- Everything else.
"""


def router_node(state: ChatState):
    user_message = next(
        (
            msg
            for msg in reversed(state["messages"])
            if isinstance(msg, HumanMessage)
        ),
        None,
    )

    if not user_message:
        return {"intent": "general"}

    if not state.get("repo_url"):
        return {"intent": "general"}

    result = router_model.invoke(
        [
            SystemMessage(content=ROUTER_PROMPT),
            user_message,
        ]
    )

    intent = result.content.strip().lower()

    return {
        "intent": "repo" if "repo" in intent else "general"
    }


def route_message(state: ChatState):
    if state["intent"] == "repo":
        return "repo_node"

    return "llm_node"


def llm_node(state: ChatState):
    response = model_with_tools.invoke(
        state["messages"]
    )

    if response.tool_calls:
        tool_results = []

        for call in response.tool_calls:
            selected_tool = next(
                t for t in tools
                if t.name == call["name"]
            )

            result = selected_tool.invoke(call["args"])

            tool_results.append({
                "role": "tool",
                "content": str(result),
                "tool_call_id": call["id"],
            })

        final_response = model_with_tools.invoke(
            state["messages"]
            + [response]
            + tool_results
        )

        return {
            "messages": [
                response,
                *tool_results,
                final_response,
            ]
        }

    return {
        "messages": [response]
    }


def repo_node(state: ChatState):
    user_message = next(
        msg
        for msg in reversed(state["messages"])
        if isinstance(msg, HumanMessage)
    )

    result = run_repo_crew(
        repo_url=state["repo_url"],
        repo_name=state["repo_name"],
        question=user_message.content,
    )

    return {
        "messages": [
            AIMessage(content=str(result))
        ]
    }


def response_node(state: ChatState):
    response = state["messages"][-1]

    save_message(
        state["thread_id"],
        "assistant",
        response.content
    )

    return {
        "final_response": response.content
    }


graph = StateGraph(ChatState)

graph.add_node("input", input_node)
graph.add_node("memory", memory_node)
graph.add_node("router", router_node)
graph.add_node("llm", llm_node)
graph.add_node("repo", repo_node)
graph.add_node("response", response_node)

graph.add_edge(START, "input")
graph.add_edge("input", "memory")
graph.add_edge("memory", "router")

graph.add_conditional_edges(
    "router",
    route_message,
    {
        "llm_node": "llm",
        "repo_node": "repo",
    },
)

graph.add_edge("llm", "response")
graph.add_edge("repo", "response")
graph.add_edge("response", END)

chatbot = graph.compile()
