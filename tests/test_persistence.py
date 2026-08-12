import pytest
from unittest.mock import patch, MagicMock
from src.persistence import save_message, load_thread_messages, get_all_threads


@patch("src.persistence._get_backend")
def test_save_message(mock_backend):
    mock_backend.return_value.save_message = MagicMock()
    save_message("t1", "user", "hello")
    mock_backend.return_value.save_message.assert_called_once_with("t1", "user", "hello")


@patch("src.persistence._get_backend")
def test_load_thread_messages(mock_backend):
    mock_backend.return_value.load_thread_messages.return_value = [
        {"role": "user", "content": "hi"}
    ]
    result = load_thread_messages("t1")
    assert len(result) == 1
    assert result[0]["role"] == "user"


@patch("src.persistence._get_backend")
def test_get_all_threads(mock_backend):
    mock_backend.return_value.get_all_threads.return_value = ["t1", "t2"]
    result = get_all_threads()
    assert result == ["t1", "t2"]
