import os
from typing import TypedDict, Annotated
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from src.config import Config
from src.vector_store import search_similar

if Config.LANGCHAIN_TRACING_V2.lower() == "true":
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = Config.LANGCHAIN_API_KEY or ""
    os.environ["LANGCHAIN_PROJECT"] = Config.LANGCHAIN_PROJECT

model = ChatOpenAI(
    model="nvidia/nemotron-3-nano-30b-a3b:free",
    base_url="https://openrouter.ai/api/v1",
    api_key=Config.OPENROUTER_API_KEY
)

search_tool = DuckDuckGoSearchRun(region="us-en")

@tool
def search_repo_knowledge(query: str) -> str:
    """Search ingested repository code and documentation to answer technical questions."""
    results = search_similar(query, top_k=5)
    if not results:
        return "No relevant repository context found."
    return "\n---\n".join([
        f"File: {r['metadata'].get('file_path', 'unknown')}\n{r['content']}"
        for r in results
    ])

tools = [search_tool, search_repo_knowledge]
model_with_tool = model.bind_tools(tools)

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState):
    messages = state["messages"]
    response = model_with_tool.invoke(messages)
    return {"messages": [response]}

tool_node = ToolNode(tools)

checkpointer = InMemorySaver()
graph = StateGraph(ChatState)

graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)
graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")

chatbot = graph.compile(checkpointer=checkpointer)