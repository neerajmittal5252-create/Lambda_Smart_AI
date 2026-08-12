import os
import sys

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../..")
    )
)

import streamlit as st
import uuid

from langchain_core.messages import HumanMessage

from src.chat_graph_fixed import chatbot
from src.persistence import load_thread_messages, get_all_threads
from src.ingest import clone_repo, chunk_and_ingest


def generate_thread_id():
    return str(uuid.uuid4())


def reset_chat():
    tid = generate_thread_id()
    st.session_state["thread_id"] = tid
    add_thread(tid)
    st.session_state["message_history"] = []


def add_thread(tid):
    if tid not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(tid)


if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = []

if "repo_url" not in st.session_state:
    st.session_state["repo_url"] = None

if "repo_name" not in st.session_state:
    st.session_state["repo_name"] = None


add_thread(st.session_state["thread_id"])


st.title("Lambda Chat AI")
st.sidebar.title("Lambda AI")


if st.sidebar.button("New Chat"):
    reset_chat()


st.sidebar.header("My Conversations")

for tid in get_all_threads():
    label = tid[:8] + "..."

    if st.sidebar.button(label, key=tid):
        st.session_state["thread_id"] = tid
        st.session_state["message_history"] = load_thread_messages(tid)


st.sidebar.divider()
st.sidebar.header("Repository RAG")


repo_url_input = st.sidebar.text_input(
    "GitHub URL",
    placeholder="https://github.com/..."
)

repo_name_input = st.sidebar.text_input(
    "Repo Name",
    placeholder="my-repo"
)


if st.sidebar.button("Ingest Repo"):
    if repo_url_input and repo_name_input:
        with st.spinner("Cloning & embedding..."):
            try:
                path = clone_repo(
                    repo_url_input,
                    repo_name_input
                )

                count = chunk_and_ingest(
                    path,
                    repo_name_input
                )

                st.session_state["repo_url"] = repo_url_input
                st.session_state["repo_name"] = repo_name_input

                st.sidebar.success(
                    f"Ingested {count} files. "
                    f"Now ask about '{repo_name_input}' in chat."
                )

            except Exception as e:
                st.sidebar.error(f"Error: {e}")

    else:
        st.sidebar.warning("Enter URL and name")


if st.session_state["repo_name"]:
    st.sidebar.caption(
        f"Active repo: {st.session_state['repo_name']}"
    )


for msg in st.session_state["message_history"]:
    if msg.get("content"):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])


user_message = st.chat_input("Type Here:")


if user_message:
    st.session_state["message_history"].append(
        {
            "role": "user",
            "content": user_message
        }
    )

    with st.chat_message("user"):
        st.markdown(user_message)

    result = chatbot.invoke(
        {
            "messages": [
                HumanMessage(content=user_message)
            ],
            "thread_id": st.session_state["thread_id"],
            "repo_url": st.session_state["repo_url"],
            "repo_name": st.session_state["repo_name"],
        }
    )

    response = result["final_response"]

    with st.chat_message("assistant"):
        st.markdown(response)

    st.session_state["message_history"].append(
        {
            "role": "assistant",
            "content": response
        }
    )
