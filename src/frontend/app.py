import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import streamlit as st
import uuid
from langchain_core.messages import HumanMessage, AIMessageChunk
from src.chat_graph import chatbot
from src.persistence import save_message, load_thread_messages, get_all_threads
from src.ingest import clone_repo, chunk_and_ingest
from src.crew.repo_crew import run_repo_crew

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

def load_conversation(tid):
    return load_thread_messages(tid)

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []
if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()
if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = []

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
        st.session_state["message_history"] = load_conversation(tid)

st.sidebar.divider()
st.sidebar.header("Repository RAG")

repo_url = st.sidebar.text_input("GitHub URL", placeholder="https://github.com/...")
repo_name = st.sidebar.text_input("Repo Name", placeholder="my-repo")

col1, col2 = st.sidebar.columns(2)
with col1:
    if st.button("Quick Ingest", key="quick_ingest"):
        if repo_url and repo_name:
            with st.spinner("Cloning & embedding..."):
                try:
                    path = clone_repo(repo_url, repo_name)
                    count = chunk_and_ingest(path, repo_name)
                    st.sidebar.success(f"Ingested {count} files")
                except Exception as e:
                    st.sidebar.error(f"Error: {e}")
        else:
            st.sidebar.warning("Enter URL and name")

with col2:
    if st.button("Crew Ingest", key="crew_ingest"):
        if repo_url and repo_name:
            with st.spinner("Running CrewAI..."):
                try:
                    result = run_repo_crew(repo_url, repo_name)
                    st.sidebar.success("Crew finished")
                    st.sidebar.markdown(f"**Summary:** {result}")
                except Exception as e:
                    st.sidebar.error(f"Error: {e}")
        else:
            st.sidebar.warning("Enter URL and name")

for msg in st.session_state["message_history"]:
    if msg.get("content"):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

user_message = st.chat_input("Type Here:")
if user_message:
    st.session_state["message_history"].append({"role": "user", "content": user_message})
    with st.chat_message("user"):
        st.markdown(user_message)

    save_message(st.session_state["thread_id"], "user", user_message)

    config = {"configurable": {"thread_id": str(st.session_state["thread_id"])}}

    with st.chat_message("assistant"):
        response = st.write_stream(
            chunk.content
            for chunk, metadata in chatbot.stream(
                {"messages": [HumanMessage(content=user_message)]},
                config=config,
                stream_mode="messages",
            )
            if isinstance(chunk, AIMessageChunk) and chunk.content
        )

    save_message(st.session_state["thread_id"], "assistant", response)
    st.session_state["message_history"].append({"role": "assistant", "content": response})
