import pytest
from unittest.mock import patch, MagicMock
from src.chat_graph import chatbot, chat_node

def test_graph_has_required_nodes():
    assert "chat_node" in chatbot.nodes
    assert "tools" in chatbot.nodes

@patch("src.chat_graph.model_with_tool")
def test_chat_node_returns_message(mock_model):
    mock_msg = MagicMock()
    mock_msg.content = "Test reply"
    mock_model.invoke.return_value = mock_msg
    result = chat_node({"messages": []})
    assert "messages" in result
    assert len(result["messages"]) == 1