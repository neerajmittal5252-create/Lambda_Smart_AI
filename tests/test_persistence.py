import pytest
from unittest.mock import patch, MagicMock
from src.persistence import save_message, load_thread_messages, get_all_threads

@patch("src.persistence.supabase")
def test_save_message(mock_supabase):
    save_message("t1", "user", "hello")
    mock_supabase.table.assert_called_with("chat_messages")
    mock_supabase.table.return_value.insert.assert_called_once()

@patch("src.persistence.supabase")
def test_load_thread_messages(mock_supabase):
    mock_supabase.table.return_value.select.return_value.eq.return_value.order.return_value.execute.return_value.data = [
        {"role": "user", "content": "hi"}
    ]
    result = load_thread_messages("t1")
    assert len(result) == 1
    assert result[0]["role"] == "user"

@patch("src.persistence.supabase")
def test_get_all_threads(mock_supabase):
    mock_supabase.table.return_value.select.return_value.order.return_value.execute.return_value.data = [
        {"thread_id": "t1"}, {"thread_id": "t1"}, {"thread_id": "t2"}
    ]
    result = get_all_threads()
    assert result==["t1","t2"]