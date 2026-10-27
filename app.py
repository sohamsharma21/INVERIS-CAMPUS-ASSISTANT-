"""Streamlit chat and password-protected administration interface."""

from __future__ import annotations

import hmac
import json
import logging
from datetime import datetime
from pathlib import Path

import streamlit as st

from invertis_bot.config import get_settings
from invertis_bot.exceptions import IndexNotFoundError, LLMError
from invertis_bot.feedback import FeedbackStore
from invertis_bot.rag import ConversationMemory, RAGEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
SETTINGS = get_settings()
CATEGORIES = ("all", "admission", "courses", "fees", "hostel", "placement", "exams", "general")

st.set_page_config(page_title="Invertis Campus Assistant", page_icon="🎓", layout="wide")
st.title("🎓 Invertis Campus Assistant")
st.caption("A source-grounded assistant for university information")
st.info("Information is from public university sources; please verify important details with the university.")
st.session_state.setdefault("chat_messages", [])
st.session_state.setdefault("memory", ConversationMemory(max_turns=SETTINGS.memory_turns))


def get_engine() -> RAGEngine:
    """Keep one RAG engine per Streamlit user session."""
    if "rag_engine" not in st.session_state:
        st.session_state.rag_engine = RAGEngine()
        st.session_state.setdefault("memory", ConversationMemory(max_turns=SETTINGS.memory_turns))
    return st.session_state.rag_engine


def last_update() -> str:
    """Show the scrape timestamp from the local raw JSON when available."""
    source_file = SETTINGS.raw_dir / "university_pages.json"
    if not source_file.exists():
        return "Not indexed yet"
    try:
        records = json.loads(source_file.read_text(encoding="utf-8"))
        timestamps = [entry.get("scraped_at", "") for entry in records if entry.get("scraped_at")]
        if timestamps:
            return max(timestamps)
        return datetime.fromtimestamp(source_file.stat().st_mtime).astimezone().isoformat(timespec="minutes")
    except (OSError, ValueError):
        return "Unknown"


def show_sources(sources: list[dict[str, object]]) -> None:
    """Render citations below the corresponding assistant message."""
    if not sources:
        st.caption("No source passages were available for this response.")
        return
    with st.expander("Sources", expanded=True):
        for source in sources:
            title = str(source["title"])
            location = str(source["location"])
            page = source.get("page")
            category = str(source["category"])
            label = f"{title} — {category}" + (f", page {page}" if page else "")
            st.markdown(f"- [{label}]({location})")


def admin_area() -> None:
    """Guard administrative actions with an environment-configured password."""
    st.subheader("Administration")
    if not SETTINGS.admin_password or not SETTINGS.admin_password.get_secret_value():
        st.warning("Set a non-empty ADMIN_PASSWORD in your local .env file to enable admin tools.")
        return
    password = st.text_input("Admin password", type="password", key="admin_password_input")
    if not hmac.compare_digest(password, SETTINGS.admin_password.get_secret_value()):
        st.caption("Enter the configured password to unlock administration.")
        return

    upload = st.file_uploader("Upload PDF, TXT, or Markdown documents", type=["pdf", "txt", "md"], accept_multiple_files=True)
    if upload and st.button("Save uploaded documents", type="primary"):
        SETTINGS.documents_dir.mkdir(parents=True, exist_ok=True)
        for item in upload:
            target = SETTINGS.documents_dir / Path(item.name).name
            target.write_bytes(item.getvalue())
        st.success(f"Saved {len(upload)} document(s). Rebuild the index to include them.")

    if st.button("Trigger re-indexing"):
        from invertis_bot.ingest import build_index
        try:
            with st.spinner("Crawling allowed pages and rebuilding the vector index…"):
                count = build_index(rebuild=True, scrape=True)
            st.success(f"Index rebuilt with {count} chunks.")
            st.session_state.pop("rag_engine", None)
        except Exception as exc:
            st.error(f"Re-indexing failed: {exc}")

    store = FeedbackStore(SETTINGS.feedback_db)
    feedback = store.recent_feedback()
    st.markdown("#### Recent feedback")
    st.dataframe([entry.__dict__ if hasattr(entry, "__dict__") else {
        "question": entry.question, "rating": entry.rating,
        "created_at": entry.created_at, "answer_excerpt": entry.answer_excerpt,
    } for entry in feedback], use_container_width=True)
    st.markdown("#### Most asked questions")
    st.dataframe([{"question": question, "count": count} for question, count in store.most_asked()], use_container_width=True)


with st.sidebar:
    st.header("Explore")
    st.caption("Answer mode: source-only (Gemini generation disabled)" if not SETTINGS.enable_gemini else f"Answer mode: Gemini ({SETTINGS.chat_model})")
    st.caption(f"Last data update: {last_update()}")
    selected_category = st.selectbox("Filter by category", CATEGORIES, index=0)
    st.markdown("#### Quick questions")
    quick_questions = {
        "Admission": "How can I apply for admission?",
        "Courses": "What programmes are offered?",
        "Hostel": "What hostel information is available?",
        "Placements": "What placement support is provided?",
        "Contact": "How can I contact the university?",
    }
    for label, question in quick_questions.items():
        if st.button(label, use_container_width=True):
            st.session_state.pending_question = question
    if st.button("Clear chat", use_container_width=True):
        st.session_state.chat_messages = []
        st.session_state.memory.clear()
        st.rerun()

chat_tab, admin_tab = st.tabs(["💬 Chat", "🔐 Admin"])
with chat_tab:
    for message_index, message in enumerate(st.session_state.chat_messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                show_sources(message.get("sources", []))
                question = message.get("question", "")
                left, right = st.columns(2)
                for column, rating, symbol in ((left, "up", "👍 Helpful"), (right, "down", "👎 Not helpful")):
                    if column.button(symbol, key=f"feedback-{message_index}-{rating}"):
                        FeedbackStore(SETTINGS.feedback_db).add_feedback(question, message["content"], rating)
                        st.toast("Thanks for your feedback!")

    pending = st.session_state.pop("pending_question", None)
    prompt = st.chat_input("Ask about admissions, courses, fees, hostel, placements, exams…")
    question = pending or prompt
    if question:
        st.session_state.chat_messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            try:
                engine = get_engine()
                prepared = engine.prepare(question, category=None if selected_category == "all" else selected_category, memory=st.session_state.memory)
                answer_text = st.write_stream(prepared.tokens)
                source_data = [
                    {"title": source.title, "location": source.location, "category": source.category, "page": source.page}
                    for source in prepared.sources
                ]
                show_sources(source_data)
                st.session_state.chat_messages.append({
                    "role": "assistant", "content": str(answer_text), "sources": source_data, "question": question,
                })
                FeedbackStore(SETTINGS.feedback_db).log_question(question)
                st.rerun()
            except (IndexNotFoundError, LLMError) as exc:
                st.error(f"The assistant is not ready yet: {exc}")
            except Exception as exc:
                logging.exception("Chat request failed")
                st.error(f"Sorry, an unexpected error occurred: {exc}")

with admin_tab:
    admin_area()
