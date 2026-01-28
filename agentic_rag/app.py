import os
import sys
import uuid
import time
import tempfile
import subprocess
import streamlit as st

from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.document_loaders import PyPDFLoader

from vectorstore import HybridRetriever
from rag_agent import build_agent
from kb_manager import cleanup_kbs
from memory import summarize_conversation, get_recent_messages
from sqlite_store import (
    init_db,
    save_message,
    load_conversation,
    save_feedback,
)

# ---------------------------------
# UTF-8 + DB Init
# ---------------------------------
os.environ["PYTHONUTF8"] = "1"
init_db()

# ---------------------------------
# Page config
# ---------------------------------
st.set_page_config(
    page_title="Agentic RAG",
    page_icon="🧠",
    layout="wide",
)

# ---------------------------------
# Claude-like Global CSS
# ---------------------------------
st.markdown("""
<style>
/* App background */
.stApp {
    background-color: #1E1F23;
    color: #E6E7EB;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #26272B;
}

/* Chat bubbles */
.chat-bubble-user {
    background-color: #32343A;
    padding: 12px 16px;
    border-radius: 14px;
    margin-bottom: 8px;
}

.chat-bubble-ai {
    background-color: #2B2D33;
    padding: 14px 18px;
    border-radius: 16px;
    margin-bottom: 10px;
    border-left: 4px solid #4F6EF7;
}

/* Buttons */
.stButton > button {
    border-radius: 999px;
    padding: 0.45rem 1.1rem;
    background-color: #1F2A44;
    color: #E6E7EB;
    border: 1px solid #4F6EF7;
    font-weight: 500;
}

.stButton > button:hover {
    background-color: #4F6EF7;
    color: #FFFFFF;
}

/* Inputs */
input, textarea {
    background-color: #26272B !important;
    color: #E6E7EB !important;
    border-radius: 10px !important;
    border: 1px solid #3A3C42 !important;
}

/* Expander (tool calls) */
.streamlit-expanderHeader {
    background-color: #26272B;
    color: #B8BCC8;
}

/* Scrollbar (optional polish) */
::-webkit-scrollbar {
    width: 6px;
}
::-webkit-scrollbar-thumb {
    background: #4F6EF7;
    border-radius: 6px;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------
# Session State
# ---------------------------------
if "conversations" not in st.session_state:
    st.session_state.conversations = {}

if "active_chat" not in st.session_state:
    cid = str(uuid.uuid4())
    st.session_state.active_chat = cid
    st.session_state.conversations[cid] = []

if "knowledge_bases" not in st.session_state:
    st.session_state.knowledge_bases = {}

if "active_kb" not in st.session_state:
    st.session_state.active_kb = None

if "agent" not in st.session_state:
    st.session_state.agent = None

if "conversation_summary" not in st.session_state:
    st.session_state.conversation_summary = ""

# ---------------------------------
# Sidebar (Claude-style minimal)
# ---------------------------------
with st.sidebar:
    st.markdown("## 💬 Conversations")

    if st.button("➕ New chat"):
        cid = str(uuid.uuid4())
        st.session_state.conversations[cid] = []
        st.session_state.active_chat = cid
        st.rerun()

    for cid in list(st.session_state.conversations.keys()):
        if st.button(cid[:8], key=f"chat-{cid}"):
            st.session_state.active_chat = cid
            if not st.session_state.conversations[cid]:
                st.session_state.conversations[cid] = load_conversation(cid)
            st.rerun()

    st.markdown("---")
    st.markdown("## 📚 Knowledge")

    source_type = st.radio("", ["PDF", "Website"], horizontal=True)

    docs = None
    source_desc = None

    if source_type == "PDF":
        pdf = st.file_uploader("Upload PDF", type=["pdf"])
        if pdf:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(pdf.read())
                docs = PyPDFLoader(tmp.name).load()
                source_desc = f"PDF: {pdf.name}"

    else:
        url = st.text_input("Website URL")
        if url:
            proc = subprocess.run(
                [sys.executable, "crawl_ingest.py", url],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            if proc.returncode == 0 and proc.stdout.strip():
                docs = [proc.stdout]
                source_desc = f"Website: {url}"
            else:
                st.error("Crawl failed")

    if docs:
        with st.spinner("Building knowledge…"):
            embeddings = OllamaEmbeddings(model="nomic-embed-text")
            retriever = HybridRetriever(docs, embeddings)

            kb_id = str(uuid.uuid4())
            st.session_state.knowledge_bases[kb_id] = {
                "retriever": retriever,
                "source": source_desc,
                "last_used": time.time(),
            }
            st.session_state.active_kb = kb_id
            st.session_state.agent = build_agent(retriever, source_desc)

        st.success("Knowledge ready")

    if st.button("🧹 Clean unused KBs"):
        cleanup_kbs(st.session_state.knowledge_bases)

# ---------------------------------
# Main Chat Area
# ---------------------------------
st.markdown(
    """
    <h2 style="
        color:#E6E7EB;
        font-weight:600;
        letter-spacing:0.6px;
        display:flex;
        align-items:center;
        gap:10px;
    ">
        <span style="color:#FF8FB1;">🧠</span>
        Agentic RAG
    </h2>
    """,
    unsafe_allow_html=True
)


messages = st.session_state.conversations[st.session_state.active_chat]

for msg in messages:
    if msg["role"] == "user":
        st.markdown(f"<div class='chat-bubble-user'>{msg['content']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='chat-bubble-ai'>{msg['content']}</div>", unsafe_allow_html=True)
        if msg.get("tool_calls"):
            with st.expander("🛠 Tool calls"):
                st.json(eval(msg["tool_calls"]))

# ---------------------------------
# Chat Input
# ---------------------------------
if st.session_state.agent:
    user_input = st.chat_input("Ask anything…")

    if user_input:
        messages.append({"role": "user", "content": user_input})
        save_message(st.session_state.active_chat, "user", user_input)

        recent = get_recent_messages(messages, max_turns=6)

        with st.markdown("<div class='loading'>Thinking<span>.</span><span>.</span><span>.</span></div>", unsafe_allow_html=True):
            result = st.session_state.agent.invoke(
                {"messages": recent, "question": user_input, "retries": 0}
            )

        answer = result["answer"]

        messages.append({"role": "assistant", "content": answer})
        save_message(
            st.session_state.active_chat,
            "assistant",
            answer,
            tool_calls=str(result.get("tool_trace")),
        )

        st.rerun()
else:
    st.info("Add a PDF or website to begin.")
