import pytest
from unittest.mock import patch, MagicMock
from src.chat_graph import chatbot, llm_node, tool_node, router_node, memory_node
from langchain_core.messages import HumanMessage, AIMessage


def test_graph_compiles():
    """Ensure the compiled graph has the expected nodes."""
    nodes = ["input_node", "memory_node", "router_node", "llm_node", "tool_node", "response_node"]
    for node in nodes:
        assert node in chatbot.nodes, f"Missing node: {node}"


def test_router_node_detects_repo_intent():
    state = {
        "messages": [HumanMessage(content="Summarize https://github.com/user/repo")],
        "thread_id": "t1",
        "repo_url": None,
        "repo_name": None,
        "intent": None,
        "final_response": None,
    }
    result = router_node(state)
    assert result.get("intent") == "repo"
    assert result.get("repo_name") == "repo"


def test_router_node_general_intent():
    state = {
        "messages": [HumanMessage(content="Hello, how are you?")],
        "thread_id": "t1",
        "repo_url": None,
        "repo_name": None,
        "intent": None,
        "final_response": None,
    }
    result = router_node(state)
    assert result.get("intent") == "general"


@patch("src.chat_graph.model_with_tools")
def test_llm_node_returns_message(mock_model):
    mock_msg = MagicMock()
    mock_msg.content = "Test reply"
    mock_msg.tool_calls = None
    mock_model.invoke.return_value = mock_msg

    result = llm_node({"messages": [HumanMessage(content="hi")]})
    assert "messages" in result
    assert len(result["messages"]) == 1


@patch("src.chat_graph.run_repo_crew")
def test_tool_node_calls_crew(mock_crew):
    mock_crew.return_value = "Crew result"
    state = {
        "messages": [HumanMessage(content="Explain this repo")],
        "thread_id": "t1",
        "repo_url": "https://github.com/user/repo",
        "repo_name": "repo",
        "intent": "repo",
        "final_response": None,
    }
    result = tool_node(state)
    assert "messages" in result
    mock_crew.assert_called_once_with(repo_name="repo", question="Explain this repo")
