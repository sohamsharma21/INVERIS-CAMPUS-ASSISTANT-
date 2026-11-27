"""Streamlit chat and password-protected administration interface."""

from __future__ import annotations

import hmac
import json
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="Invertis Campus Assistant", page_icon="🎓", layout="wide")

try:
    cloud_secrets = st.secrets
except st.errors.StreamlitSecretNotFoundError:
    cloud_secrets = {}

for secret_name in (
    "GOOGLE_API_KEY",
    "ADMIN_PASSWORD",
    "ENABLE_GEMINI",
    "CHAT_MODEL",
    "EMBEDDING_MODEL",
    "WEBSITE_URL",
    "ENABLE_PDF_OCR",
    "OCR_LANGUAGE",
    "ENABLE_CRAWL_SCHEDULE",
    "CRAWL_SCHEDULE_INTERVAL_HOURS",
):
    if secret_name in cloud_secrets:
        os.environ.setdefault(secret_name, str(cloud_secrets[secret_name]))

from invertis_bot.config import get_settings
from invertis_bot.exceptions import IndexNotFoundError, LLMError
from invertis_bot.feedback import FeedbackStore
from invertis_bot.rag import ConversationMemory, RAGEngine
from invertis_bot.ingest import read_schedule_status, write_schedule_status

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
SETTINGS = get_settings()
CATEGORIES = ("all", "admission", "courses", "fees", "hostel", "placement", "exams", "general")
st.session_state.setdefault("appearance", "Dark")

st.markdown(
    """
    <style>
    :root {
        --brand: #5b5bd6;
        --brand-dark: #4545a9;
        --surface: #f7f8fc;
        --border: rgba(49, 51, 63, 0.10);
    }
    section { scroll-behavior: smooth; }
    [data-testid="stAppViewContainer"] {
        background:
            radial-gradient(circle at 82% 12%, rgba(91, 91, 214, 0.13), transparent 24rem),
            radial-gradient(circle at 18% 82%, rgba(56, 189, 248, 0.10), transparent 28rem),
            linear-gradient(145deg, #f8fafc 0%, #eef2ff 48%, #e0f2fe 100%);
        background-attachment: fixed;
        color: #0f172a;
    }
    [data-testid="stAppViewContainer"] p,
    [data-testid="stAppViewContainer"] label,
    [data-testid="stAppViewContainer"] small,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] small,
    [data-testid="stCaptionContainer"] {
        color: #334155 !important;
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4 {
        color: #1e293b !important;
    }
    [data-testid="stSidebar"] button,
    [data-testid="stSidebar"] button p {
        color: #1e293b !important;
    }
    [data-testid="stChatInput"] textarea {
        color: #0f172a !important;
        background: rgba(255, 255, 255, 0.96) !important;
    }
    [data-testid="stChatInput"] textarea::placeholder {
        color: #64748b !important;
    }
    .block-container {
        max-width: 1180px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }
    h1 {
        background: linear-gradient(90deg, #ffffff, #a5b4fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.035em;
    }
    [data-testid="stChatMessage"] {
        border: 1px solid rgba(51, 65, 85, 0.12);
        border-radius: 18px;
        padding: 0.65rem 0.85rem;
        box-shadow: 0 10px 28px rgba(40, 44, 92, 0.10);
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        background: rgba(255, 255, 255, 0.82);
        color: #0f172a;
    }
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIconAssistant"]) {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.10), rgba(255, 255, 255, 0.92));
        border-left: 3px solid #6366f1;
    }
    [data-testid="stChatInput"] textarea {
        border-radius: 16px;
        border: 1px solid rgba(91, 91, 214, 0.28);
        box-shadow: 0 8px 28px rgba(49, 51, 63, 0.08);
    }
    [data-testid="stSidebar"] {
        background: rgba(248, 250, 252, 0.88);
        backdrop-filter: blur(18px);
        -webkit-backdrop-filter: blur(18px);
        border-right: 1px solid rgba(51, 65, 85, 0.10);
        color: #0f172a;
    }
    [data-testid="stSidebar"] button {
        border-radius: 10px;
        border: 1px solid rgba(91, 91, 214, 0.12);
        transition: 0.18s ease;
    }
    [data-testid="stSidebar"] button:hover {
        background: var(--brand);
        color: white;
        transform: translateY(-1px);
    }
    [data-testid="stTabs"] [role="tab"][aria-selected="true"] {
        color: var(--brand-dark);
        border-bottom-color: var(--brand);
    }
    [data-testid="stAlert"] {
        border-radius: 12px;
        border: 1px solid rgba(165, 180, 252, 0.18);
        background: rgba(30, 41, 59, 0.82);
        color: #f8fafc;
    }
    .welcome-card {
        border: 1px solid rgba(165, 180, 252, 0.20);
        border-radius: 18px;
        padding: 1.15rem 1.35rem;
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.28), rgba(15, 23, 42, 0.82));
        color: #f8fafc;
        box-shadow: 0 10px 30px rgba(40, 44, 92, 0.06);
    }
    [data-testid="stExpander"] {
        border: 1px solid rgba(49, 51, 63, 0.12);
        border-radius: 12px;
    }
    [data-testid="stChatInput"] {
        max-width: 1180px;
        margin: 0 auto;
    }
    @media (max-width: 768px) {
        .block-container {
            max-width: 100%;
            padding: 1rem 0.75rem 5.5rem 0.75rem;
        }
        h1 {
            font-size: 1.75rem !important;
            line-height: 1.2;
        }
        [data-testid="stCaptionContainer"] {
            font-size: 0.82rem;
        }
        [data-testid="stChatMessage"] {
            border-radius: 12px;
            padding: 0.35rem 0.55rem;
            margin-bottom: 0.4rem;
        }
        [data-testid="stChatInput"] {
            position: sticky;
            bottom: 0.25rem;
            z-index: 20;
            padding: 0.25rem;
            background: var(--background-color);
        }
        [data-testid="stChatInput"] textarea {
            min-height: 48px;
            font-size: 16px;
        }
        [data-testid="column"] {
            width: 100% !important;
            flex: 1 1 100% !important;
        }
        [data-testid="stTabs"] button {
            flex: 1 1 auto;
            min-width: 0;
        }
        [data-testid="stDataFrame"] {
            overflow-x: auto;
        }
        .welcome-card {
            border-radius: 12px;
            padding: 0.9rem 1rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)
if st.session_state.appearance == "Dark":
    st.markdown(
        """
        <style>
        [data-testid="stAppViewContainer"] {
            background: radial-gradient(circle at 80% 10%, rgba(129, 140, 248, .20), transparent 24rem),
                        linear-gradient(145deg, #0b1224 0%, #111c3b 52%, #25245f 100%) !important;
            color: #f8fafc !important;
        }
        [data-testid="stHeader"], [data-testid="stToolbar"] { background: transparent !important; }
        [data-testid="stSidebar"] {
            background: rgba(10, 18, 36, .96) !important;
            border-right: 1px solid rgba(165, 180, 252, .14) !important;
        }
        [data-testid="stAppViewContainer"] p,
        [data-testid="stAppViewContainer"] span,
        [data-testid="stAppViewContainer"] label,
        [data-testid="stAppViewContainer"] small,
        [data-testid="stCaptionContainer"],
        [data-testid="stMarkdownContainer"],
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] h4 {
            color: #e8edf7 !important;
        }
        [data-testid="stSidebar"] button,
        [data-testid="stSidebar"] button p {
            color: #f8fafc !important;
            background: rgba(30, 41, 59, .78) !important;
        }
        [data-testid="stChatMessage"],
        [data-testid="stExpander"],
        [data-testid="stAlert"] {
            background: rgba(20, 31, 55, .92) !important;
            color: #f8fafc !important;
            border: 1px solid rgba(165, 180, 252, .18) !important;
        }
        [data-testid="stChatMessage"] p,
        [data-testid="stChatMessage"] span,
        [data-testid="stExpander"] p,
        [data-testid="stExpander"] strong,
        [data-testid="stAlert"] p,
        [data-testid="stAlert"] span,
        .welcome-card,
        .welcome-card strong {
            color: #f8fafc !important;
        }
        .welcome-card {
            background: linear-gradient(135deg, rgba(99,102,241,.30), rgba(20,31,55,.92)) !important;
            border-color: rgba(165,180,252,.22) !important;
        }
        [data-testid="stChatInput"] textarea {
            background: #151f35 !important;
            color: #f8fafc !important;
            border: 1px solid rgba(165, 180, 252, .32) !important;
        }
        [data-testid="stChatInput"] textarea::placeholder { color: #aab5c8 !important; }
        [data-testid="stTabs"] button, [data-testid="stTabs"] button p { color: #cbd5e1 !important; }
        [data-testid="stTabs"] [role="tab"][aria-selected="true"],
        [data-testid="stTabs"] [role="tab"][aria-selected="true"] p { color: #fff !important; }
        h1 {
            background: linear-gradient(90deg, #fff, #b8c1ff) !important;
            -webkit-background-clip: text !important;
            -webkit-text-fill-color: transparent !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        """
        <style>
        [data-testid="stAppViewContainer"] {
            background: radial-gradient(circle at 82% 12%, rgba(91,91,214,.12), transparent 24rem),
                        linear-gradient(145deg, #f8fafc 0%, #eef2ff 48%, #e0f2fe 100%) !important;
            color: #172033 !important;
        }
        [data-testid="stSidebar"] {
            background: rgba(248,250,252,.94) !important;
            border-right: 1px solid rgba(51,65,85,.10) !important;
        }
        [data-testid="stAppViewContainer"] p,
        [data-testid="stAppViewContainer"] span,
        [data-testid="stAppViewContainer"] label,
        [data-testid="stAppViewContainer"] small,
        [data-testid="stCaptionContainer"],
        [data-testid="stMarkdownContainer"],
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] h4 {
            color: #334155 !important;
        }
        [data-testid="stChatMessage"],
        [data-testid="stExpander"],
        [data-testid="stAlert"],
        .welcome-card {
            background: rgba(255,255,255,.88) !important;
            color: #172033 !important;
            border: 1px solid rgba(51,65,85,.12) !important;
        }
        [data-testid="stChatMessage"] p,
        [data-testid="stChatMessage"] span,
        [data-testid="stExpander"] p,
        [data-testid="stAlert"] p,
        .welcome-card,
        .welcome-card strong {
            color: #172033 !important;
        }
        [data-testid="stChatInput"] textarea {
            background: #fff !important;
            color: #172033 !important;
            border: 1px solid rgba(91,91,214,.28) !important;
        }
        h1 {
            background: linear-gradient(90deg, #1e293b, #4f46e5) !important;
            -webkit-background-clip: text !important;
            -webkit-text-fill-color: transparent !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
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
            excerpt = str(source.get("excerpt", "")).strip()
            label = f"{title} — {category}" + (f", page {page}" if page else "")
            st.markdown(f"**[{label}]({location})**")
            if excerpt:
                st.caption(f"“{excerpt}”")


def admin_area() -> None:
    """Guard administrative actions with an environment-configured password."""
    st.subheader("Administration")
    if not SETTINGS.admin_password or not SETTINGS.admin_password.get_secret_value():
        st.warning("Set a non-empty ADMIN_PASSWORD in your local .env file to enable admin tools.")
        return
    if not st.session_state.get("admin_authenticated", False):
        with st.form("admin_login_form", clear_on_submit=False):
            password = st.text_input("Admin password", type="password", key="admin_password_input")
            submitted = st.form_submit_button("Enter Admin Panel", type="primary", use_container_width=True)
        if submitted:
            if hmac.compare_digest(password, SETTINGS.admin_password.get_secret_value()):
                st.session_state.admin_authenticated = True
                st.rerun()
            st.error("Incorrect admin password.")
        st.caption("Enter the configured password to unlock administration.")
        return

    if st.button("Lock admin panel", key="admin_logout"):
        st.session_state.admin_authenticated = False
        st.rerun()

    with st.expander("Change admin password", expanded=False):
        current_password = st.text_input("Current password", type="password", key="change_admin_current")
        new_password = st.text_input("New password", type="password", key="change_admin_new")
        confirm_password = st.text_input("Confirm new password", type="password", key="change_admin_confirm")
        if st.button("Update password", type="primary"):
            if not hmac.compare_digest(current_password, SETTINGS.admin_password.get_secret_value()):
                st.error("Current password is incorrect.")
            elif len(new_password) < 12:
                st.error("New password must contain at least 12 characters.")
            elif new_password != confirm_password:
                st.error("New password and confirmation do not match.")
            elif hmac.compare_digest(new_password, SETTINGS.admin_password.get_secret_value()):
                st.error("Choose a password different from the current password.")
            else:
                env_path = Path(__file__).resolve().parent / ".env"
                lines = env_path.read_text(encoding="utf-8").splitlines() if env_path.exists() else []
                prefix = "ADMIN_PASSWORD="
                replaced = False
                for index, line in enumerate(lines):
                    if line.startswith(prefix):
                        lines[index] = prefix + new_password
                        replaced = True
                        break
                if not replaced:
                    lines.append(prefix + new_password)
                env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
                st.success("Password updated. Restarting the app will activate it.")
                st.caption("After restart, use the new password to open administration.")

    dashboard_tab, scheduler_tab, content_tab, feedback_tab = st.tabs(
        ["Dashboard", "Scheduler", "Content", "Feedback"]
    )

    store = FeedbackStore(SETTINGS.feedback_db)
    feedback = store.recent_feedback()
    popular = store.most_asked()

    raw_file = SETTINGS.raw_dir / "university_pages.json"
    try:
        raw_records = json.loads(raw_file.read_text(encoding="utf-8")) if raw_file.exists() else []
    except (OSError, json.JSONDecodeError):
        raw_records = []

    document_count = sum(
        1
        for path in SETTINGS.documents_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".pdf", ".txt", ".md"}
    ) if SETTINGS.documents_dir.exists() else 0

    backend_file = SETTINGS.index_dir / ".embedding_backend"
    embedding_backend = backend_file.read_text(encoding="utf-8").strip() if backend_file.exists() else "not built"
    index_ready = (SETTINGS.index_dir / "chroma.sqlite3").exists()
    schedule_status = read_schedule_status(SETTINGS)
    schedule_last_run = str(schedule_status.get("last_success_at") or "Never")
    schedule_state = str(schedule_status.get("status") or "Not configured")
    positive = sum(1 for entry in feedback if entry.rating == "up")
    negative = sum(1 for entry in feedback if entry.rating == "down")
    total_ratings = positive + negative
    satisfaction = round((positive / total_ratings) * 100) if total_ratings else 0

    with dashboard_tab:
        st.markdown("#### System overview")
        metric_one, metric_two, metric_three, metric_four = st.columns(4)
        metric_one.metric("Index", "Ready" if index_ready else "Missing")
        metric_two.metric("Web sources", len(raw_records))
        metric_three.metric("Local documents", document_count)
        metric_four.metric("Helpful rate", f"{satisfaction}%")

        st.markdown("#### Knowledge base")
        knowledge_left, knowledge_right = st.columns(2)
        with knowledge_left:
            st.dataframe(
                [
                    {"Setting": "Embedding backend", "Value": embedding_backend},
                    {"Setting": "Index directory", "Value": str(SETTINGS.index_dir)},
                    {"Setting": "Web source file", "Value": str(raw_file)},
                    {"Setting": "Gemini", "Value": "Enabled" if SETTINGS.enable_gemini else "Disabled"},
                    {"Setting": "Scheduled crawl", "Value": "Enabled" if SETTINGS.enable_crawl_schedule else "Disabled"},
                    {"Setting": "Schedule interval", "Value": f"{SETTINGS.crawl_schedule_interval_hours:g} hours"},
                    {"Setting": "Last crawl", "Value": schedule_last_run},
                    {"Setting": "Crawl state", "Value": schedule_state},
                ],
                hide_index=True,
                use_container_width=True,
            )
        with knowledge_right:
            st.markdown("**Recent activity**")
            st.metric("Feedback entries", len(feedback))
            st.metric("Unique popular questions", len(popular))
            st.metric("Positive / negative", f"{positive} / {negative}")

    with scheduler_tab:
        st.markdown("#### Crawl scheduler")
        enabled = st.toggle(
            "Enable automatic crawling",
            value=SETTINGS.enable_crawl_schedule,
            key="crawler_schedule_enabled",
        )
        interval = st.number_input(
            "Run interval (hours)",
            min_value=1.0,
            max_value=168.0,
            value=float(SETTINGS.crawl_schedule_interval_hours),
            step=1.0,
            key="crawler_schedule_interval",
        )

        next_run = "Waiting for first successful run"
        last_success = schedule_status.get("last_success_at")
        if isinstance(last_success, str):
            try:
                last_dt = datetime.fromisoformat(last_success)
                next_dt = last_dt + timedelta(hours=float(interval))
                next_run = next_dt.astimezone().strftime("%Y-%m-%d %H:%M")
            except (TypeError, ValueError):
                next_run = "Unknown"

        schedule_metric_one, schedule_metric_two, schedule_metric_three = st.columns(3)
        schedule_metric_one.metric("Scheduler", "Enabled" if enabled else "Disabled")
        schedule_metric_two.metric("Last successful run", schedule_last_run)
        schedule_metric_three.metric("Next run", next_run)

        if st.button("Save scheduler settings", type="primary"):
            env_path = SETTINGS.data_dir.parent / ".env"
            lines = env_path.read_text(encoding="utf-8").splitlines() if env_path.exists() else []
            replacements = {
                "ENABLE_CRAWL_SCHEDULE": "true" if enabled else "false",
                "CRAWL_SCHEDULE_INTERVAL_HOURS": str(float(interval)),
            }
            for key, value in replacements.items():
                prefix = f"{key}="
                for index, line in enumerate(lines):
                    if line.startswith(prefix):
                        lines[index] = prefix + value
                        break
                else:
                    lines.append(prefix + value)
            env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            st.success("Scheduler settings saved. Restart the app to apply runtime values.")

        if st.button("Run crawl now", type="secondary"):
            from invertis_bot.ingest import build_index
            write_schedule_status(SETTINGS, status="running")
            try:
                with st.spinner("Crawling and rebuilding the knowledge index…"):
                    count = build_index(rebuild=True, scrape=True)
                write_schedule_status(
                    SETTINGS,
                    status="success",
                    last_success_at=datetime.now(timezone.utc).isoformat(),
                    chunks=count,
                )
                st.success(f"Crawl completed: {count} chunks indexed.")
                st.session_state.pop("rag_engine", None)
            except Exception as exc:
                write_schedule_status(SETTINGS, status="failed", error=str(exc))
                st.error(f"Crawl failed: {exc}")

        st.markdown("#### Scheduler status")
        st.dataframe(
            [
                {"Field": "Status", "Value": schedule_state},
                {"Field": "Last successful run", "Value": schedule_last_run},
                {"Field": "Updated at", "Value": str(schedule_status.get("updated_at") or "Never")},
                {"Field": "Last chunk count", "Value": str(schedule_status.get("chunks") or "N/A")},
                {"Field": "Error", "Value": str(schedule_status.get("error") or "None")},
            ],
            hide_index=True,
            use_container_width=True,
        )

    with content_tab:
        st.markdown("#### Manage content")
        st.caption("Uploaded documents are indexed only after a rebuild.")
        uploaded_names = [
            path.name
            for path in sorted(SETTINGS.documents_dir.rglob("*"))
            if path.is_file() and path.suffix.lower() in {".pdf", ".txt", ".md"}
        ] if SETTINGS.documents_dir.exists() else []
        st.dataframe(
            [{"Document": name} for name in uploaded_names],
            hide_index=True,
            use_container_width=True,
        )

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

    with feedback_tab:
        st.markdown("#### Recent feedback")
        st.dataframe([{
            "question": entry.question,
            "rating": "Helpful" if entry.rating == "up" else "Not helpful",
            "created_at": entry.created_at,
            "answer_excerpt": entry.answer_excerpt,
        } for entry in feedback], hide_index=True, use_container_width=True)
        st.markdown("#### Most asked questions")
        st.dataframe(
            [{"question": question, "count": count} for question, count in popular],
            hide_index=True,
            use_container_width=True,
        )


with st.sidebar:
    st.header("Explore")
    st.selectbox("Appearance", ("Dark", "Light"), key="appearance")
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
    if not st.session_state.chat_messages:
        st.markdown(
            """
            <div class="welcome-card">
            <div style="font-size:1.08rem; font-weight:750;">Welcome to the Invertis Campus Guide</div>
            <div style="margin-top:.35rem;">Ask in English, Hindi, or Hinglish. I’ll ground answers in university sources.</div>
            <div style="margin-top:.8rem; display:flex; gap:.45rem; flex-wrap:wrap;">
              <span style="padding:.35rem .65rem; border-radius:999px; background:rgba(99,102,241,.12);">Admissions</span>
              <span style="padding:.35rem .65rem; border-radius:999px; background:rgba(99,102,241,.12);">Courses</span>
              <span style="padding:.35rem .65rem; border-radius:999px; background:rgba(99,102,241,.12);">Fees</span>
              <span style="padding:.35rem .65rem; border-radius:999px; background:rgba(99,102,241,.12);">Placements</span>
            </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
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
                    {
                        "title": source.title,
                        "location": source.location,
                        "category": source.category,
                        "page": source.page,
                        "excerpt": getattr(source, "excerpt", ""),
                    }
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
