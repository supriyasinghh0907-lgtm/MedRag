import os
import json
import html
import uuid
from datetime import datetime

import markdown as md
import streamlit as st

from medrag.config import MAX_PDFS, MAX_PDF_MB
from medrag.pdf_processor import extract_pdf_text, chunk_text
from medrag.vector_store import SessionStore, get_model
from medrag.agents import classification_agent
from medrag.rag_pipeline import run_rag_pipeline

st.set_page_config(
    page_title="MedRag - Medical Research Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Optional password: only enforced if APP_PASSWORD is set as a secret.
APP_PASSWORD = os.getenv("APP_PASSWORD", "")
if APP_PASSWORD and not st.session_state.get("authenticated"):
    st.markdown("<h2 style='text-align:center;'>🔐 MedRag Login</h2>", unsafe_allow_html=True)
    pw = st.text_input("Enter Password", type="password")
    if st.button("Login"):
        if pw == APP_PASSWORD:
            st.session_state.authenticated = True
            st.rerun()
        st.error("Incorrect password. Please try again.")
    st.stop()

with open(os.path.join(os.path.dirname(__file__), "style.css")) as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading embedding model…")
def warm_up():
    get_model()
    return True


warm_up()


def new_session():
    cid = str(uuid.uuid4())[:8]
    st.session_state.chats[cid] = {
        "messages": [],
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "title": "New Research Session",
        "store": SessionStore(),   # each session has its own private index
        "files": [],
    }
    st.session_state.current_chat = cid


def render(text):
    """Escape any HTML first, then render markdown, so documents can't inject markup."""
    return md.markdown(html.escape(text), extensions=["nl2br"])


if "chats" not in st.session_state:
    st.session_state.chats = {}
    new_session()

chat = st.session_state.chats[st.session_state.current_chat]

# ---------------- Sidebar ----------------
with st.sidebar:
    st.markdown("""
        <div class="sidebar-header">
            <div class="logo-container"><span class="logo">🏥</span><h2 class="brand-name">MedRag</h2></div>
            <p class="tagline">Multi-agent RAG for medical research</p>
        </div>""", unsafe_allow_html=True)
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    if c1.button("New Session", use_container_width=True):
        new_session()
        st.rerun()
    if c2.button("Clear All", use_container_width=True):
        st.session_state.chats = {}
        new_session()
        st.rerun()

    st.markdown("<h3 class='sidebar-section-title'>Research History</h3>", unsafe_allow_html=True)
    for cid in reversed(list(st.session_state.chats)):
        data = st.session_state.chats[cid]
        active = cid == st.session_state.current_chat
        if st.button(f"{'🔵' if active else '⚪'} {data['title'][:22]}", key=f"chat_{cid}",
                     use_container_width=True, help=f"Created: {data['created_at']}"):
            st.session_state.current_chat = cid
            st.rerun()

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    st.markdown("<h3 class='sidebar-section-title'>Session Stats</h3>", unsafe_allow_html=True)
    m1, m2 = st.columns(2)
    m1.metric("Messages", len(chat["messages"]))
    m2.metric("PDFs", len(chat["files"]))
    st.caption("Uploads stay in your browser session only and are discarded when you leave. "
               "Please don't upload real patient records.")

# ---------------- Main ----------------
st.markdown("""
    <div class="main-header">
        <h1 class="main-title"><span class="gradient-text">Medical Research Assistant</span></h1>
        <p class="subtitle">Upload medical PDFs and ask questions — answers are drafted by one agent and checked by a second</p>
    </div>""", unsafe_allow_html=True)

if not chat["files"]:
    st.markdown("""
        <div class="upload-section"><div class="upload-card">
            <div class="upload-icon">📄</div>
            <h2>Upload Medical Documents</h2>
            <p class="upload-description">Research papers, clinical reports and scientific documents. Non-medical PDFs are rejected by a classifier agent.</p>
        </div></div>""", unsafe_allow_html=True)

    uploads = st.file_uploader("Choose PDF files", type=["pdf"], accept_multiple_files=True,
                               label_visibility="collapsed")

    if st.button("No PDF handy? Try a sample paper", use_container_width=True):
        with open(os.path.join(os.path.dirname(__file__), "sample", "sample_paper.json")) as f:
            sample = json.load(f)
        with st.spinner("Indexing sample paper…"):
            chat["store"].add(sample["chunks"])
        chat["files"].append(sample["title"])
        st.rerun()

    if uploads:
        if len(uploads) > MAX_PDFS:
            st.error(f"Please upload at most {MAX_PDFS} PDFs at a time.")
            st.stop()
        progress = st.progress(0.0)
        for i, file in enumerate(uploads):
            progress.progress((i + 1) / len(uploads), text=f"Processing {file.name}…")
            data = file.getvalue()
            if len(data) > MAX_PDF_MB * 1024 * 1024:
                st.error(f"{file.name} is larger than {MAX_PDF_MB} MB.")
                continue
            try:
                text = extract_pdf_text(data)
                if not classification_agent(text):
                    st.error(f"❌ {file.name} doesn't look like a medical document (or is a scanned image).")
                    continue
                chat["store"].add(chunk_text(text))
                chat["files"].append(file.name)
            except Exception as e:
                st.error(f"Error processing {file.name}: {e}")
        progress.empty()
        if chat["files"]:
            st.rerun()

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    f1, f2, f3 = st.columns(3)
    for col, icon, title, desc in [
        (f1, "🤖", "AI-Powered Analysis", "A primary agent answers using passages retrieved from your documents"),
        (f2, "🔍", "Semantic Search", "Sentence-transformer embeddings + FAISS find the most relevant passages"),
        (f3, "✨", "Validated Answers", "A second agent reviews every answer for accuracy and safety"),
    ]:
        col.markdown(f"""<div class="feature-card"><div class="feature-icon">{icon}</div>
            <h3>{title}</h3><p>{desc}</p></div>""", unsafe_allow_html=True)

else:
    names = ", ".join(html.escape(n) for n in chat["files"][:3])
    st.markdown(f"""<div class="document-status"><span class="status-icon"></span>
        <span class="status-text">Active Documents: {names}</span></div>""", unsafe_allow_html=True)

    if not chat["messages"]:
        st.markdown("""
            <div class="empty-chat">
                <div class="empty-chat-icon">💬</div>
                <h3>Start Your Research</h3>
                <p>Ask any question about your uploaded medical documents</p>
                <div class="suggested-questions"><p class="suggestions-title">Try asking:</p>
                <ul><li>What are the key findings in this research?</li>
                <li>What methods did the study use?</li>
                <li>What limitations do the authors mention?</li></ul></div>
            </div>""", unsafe_allow_html=True)

    for msg in chat["messages"]:
        who = "user" if msg["role"] == "user" else "assistant"
        avatar = "👤" if who == "user" else "🤖"
        st.markdown(f"""
            <div class="message-container {who}-message-container">
              <div class="message {who}-message">
                <div class="message-avatar {who}-avatar">{avatar}</div>
                <div class="message-content">
                  <div class="message-text">{render(msg['content'])}</div>
                  <div class="message-time">{msg.get('timestamp', '')}</div>
                </div>
              </div>
            </div>""", unsafe_allow_html=True)
        if who == "assistant" and msg.get("trace"):
            with st.expander("How this answer was made (retrieved passages + Agent 1 draft)"):
                st.markdown("**Retrieved passages**")
                st.text(msg["trace"]["context_used"][:4000])
                st.markdown("**Agent 1 draft (before validation)**")
                st.markdown(msg["trace"]["agent1_reasoning"])

    question = st.chat_input("Type your medical question here…")
    if question:
        ts = datetime.now().strftime("%H:%M")
        chat["messages"].append({"role": "user", "content": question, "timestamp": ts})
        if len(chat["messages"]) == 1:
            chat["title"] = question[:30]
        with st.spinner("Analyzing medical documents…"):
            try:
                out = run_rag_pipeline(chat["store"], question)
                reply, trace = out["final"], out
            except Exception as e:
                reply, trace = f"Sorry, something went wrong: {e}", None
        chat["messages"].append({"role": "assistant", "content": reply, "timestamp": ts, "trace": trace})
        st.rerun()

st.markdown("""<div class="footer"><p>MedRag • For research and education only — not medical advice</p></div>""",
            unsafe_allow_html=True)
