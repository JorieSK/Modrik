from dotenv import load_dotenv
import streamlit as st
from core.ai_service import stream_response
from core.file_service import extract_text
from core.vector_store import load_from_disk, retrieve_with_scores, is_loaded
from core.pii_guard import redact_pii

load_dotenv()

st.set_page_config(
    page_title="Mudrik",
    page_icon="🔵",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Arabic:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'IBM Plex Sans Arabic', sans-serif; }

    #MainMenu, footer, header { visibility: hidden; }
    [data-testid="stSidebar"] { display: none; }
    .block-container { padding-top: 2rem; padding-bottom: 7rem; max-width: 760px; }
    [data-testid="stAppViewContainer"], body, html { background: #FAFEFE; direction: rtl; }

    /* ── Scrollbar styling only — don't touch overflow/flex ── */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: #D0F0F3; border-radius: 4px; }
    ::-webkit-scrollbar-thumb:hover { background: #B2E8EC; }

    /* ── Header ── */
    .top-header { text-align: center; padding: 0 0 1.4rem; }
    .top-header h1 { font-size: 2.4rem; font-weight: 700; color: #1A9AA8; margin: 0 0 0.2rem; }
    .top-header p  { font-size: 1rem; color: #8FC9CE; margin: 0; }
    [data-testid="stHeaderActionElements"] { display: none !important; }

    /* ── File tag ── */
    .file-tag {
        background: #E6F9FA;
        border: 1px solid #B2E8EC;
        border-radius: 8px;
        padding: 0.3rem 0.8rem;
        font-size: 0.8rem;
        color: #1A9AA8;
        display: inline-block;
    }

    /* ── Chat messages ── */
    [data-testid="stChatMessage"] { gap: 0.7rem; }
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
        background: #E6F9FA;
        border-radius: 1.1rem;
        padding: 0.75rem 1.15rem;
        max-width: 75%;
        margin-left: auto;
        margin-right: 0;
        text-align: right;
    }
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
        background: transparent;
        padding: 0.3rem 0 0.6rem;
    }
    [data-testid="stChatMessageAvatarUser"], [data-testid="stChatMessageAvatarAssistant"] {
        background: #1A9AA8 !important;
        color: #fff !important;
    }

    /* ── Chat input ── */
    [data-testid="stChatInput"] textarea {
        border: 1.5px solid #D8F1F3 !important;
        border-radius: 1.4rem !important;
        background: #FFFFFF !important;
        font-size: 0.95rem !important;
        padding: 0.85rem 1.2rem !important;
    }
    [data-testid="stChatInput"] textarea:focus {
        border-color: #4ECDD4 !important;
        box-shadow: 0 0 0 3px rgba(78,205,212,0.15) !important;
    }
    [data-testid="stChatInputFileUploadButton"],
    [data-testid="stChatInputSubmitButton"] {
        color: #1A9AA8 !important;
    }

    /* ── Sources expander scrollable ── */
    [data-testid="stExpanderDetails"] {
        max-height: 420px;
        overflow-y: auto;
        overflow-x: hidden;
    }
    [data-testid="stExpanderDetails"]::-webkit-scrollbar { width: 6px; }
    [data-testid="stExpanderDetails"]::-webkit-scrollbar-track { background: transparent; }
    [data-testid="stExpanderDetails"]::-webkit-scrollbar-thumb {
        background: #B2E8EC;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

# ── Load vector index once per session ────────────────────────
if "rag_ready" not in st.session_state:
    st.session_state.rag_ready = load_from_disk()

# ── Session state ──────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "file_text" not in st.session_state:
    st.session_state.file_text = ""
if "file_name" not in st.session_state:
    st.session_state.file_name = ""


# ── Logo + title ───────────────────────────────────────────────
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    st.image("assets/logo.png", use_container_width=True)

st.markdown("""
<div class="top-header">
    <h1>مدرك</h1>
    <p>مساعدك الذكي لفهم نظام العمل السعودي</p>
</div>
""", unsafe_allow_html=True)

# ── Helper functions ────────────────────────────────────────────
def show_sources(scored_chunks: list[tuple[str, float]], file_text: str):
    parts = []
    if scored_chunks:
        parts.append(f"{len(scored_chunks)} مواد من نظام العمل")
    if file_text:
        parts.append("ملف العقد")
    if not parts:
        return
    with st.popover(f"المصادر — {' · '.join(parts)}"):
        if file_text:
            st.markdown("**ملف العقد المرفوع**")
            preview = file_text[:2000] + ("…" if len(file_text) > 2000 else "")
            st.text_area("", value=preview, height=180, disabled=True, label_visibility="collapsed")
            if scored_chunks:
                st.divider()
        for i, (chunk, score) in enumerate(scored_chunks, 1):
            pct = int(score * 100)
            color = "#1A9AA8" if pct >= 70 else "#F39C12" if pct >= 50 else "#999"
            st.markdown(
                f"**المادة {i}** &nbsp; "
                f'<span style="font-size:0.75rem;color:{color};">تطابق {pct}%</span>',
                unsafe_allow_html=True,
            )
            st.text_area("", value=chunk, height=130, disabled=True, key=f"chunk_{i}_{hash(chunk)}", label_visibility="collapsed")
            if i < len(scored_chunks):
                st.divider()

def send(prompt: str):
    clean_prompt, pii_count = redact_pii(prompt)
    st.session_state.messages.append({"role": "user", "content": clean_prompt})
    with st.chat_message("user"):
        st.markdown(clean_prompt)
        if pii_count:
            st.caption(f"تم إخفاء {pii_count} بيانات حساسة تلقائياً")

    scored_chunks = retrieve_with_scores(clean_prompt, k=5) if is_loaded() else []
    file_text = st.session_state.file_text

    context_parts = []
    if file_text:
        context_parts.append(f"=== ملف القضية ===\n{file_text}")
    if scored_chunks:
        law_text = "\n\n---\n\n".join(chunk for chunk, _ in scored_chunks)
        context_parts.append(f"=== مواد نظام العمل ذات الصلة ===\n{law_text}")
    context = "\n\n".join(context_parts)

    with st.chat_message("assistant", avatar="✨"):
        response = st.write_stream(stream_response(prompt, context=context))

    show_sources(scored_chunks, file_text)
    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
        "scored_chunks": scored_chunks,
        "file_text": file_text,
    })

# ── Chat ───────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="✨" if msg["role"] == "assistant" else None):
        st.markdown(msg["content"])
    if msg["role"] == "assistant":
        show_sources(msg.get("scored_chunks", []), msg.get("file_text", ""))

prompt = st.chat_input(
    "اسألني عن وضعك في العقد…",
    accept_file=True,
    file_type=["txt", "pdf"],
)

if prompt and prompt.files:
    uploaded_file = prompt.files[0]
    with st.spinner("جاري قراءة الملف…"):
        raw_text = extract_text(uploaded_file)
        redacted_text, pii_count = redact_pii(raw_text)
        st.session_state.file_text = redacted_text
        st.session_state.file_name = uploaded_file.name
    if not prompt.text:
        notice = f"تم إرفاق الملف: {st.session_state.file_name}"
        if pii_count:
            notice += f"\n\nتم إخفاء {pii_count} بيانات حساسة (هوية / IBAN) تلقائياً"
        st.session_state.messages.append({"role": "user", "content": notice})
        with st.chat_message("user"):
            st.markdown(notice)

if prompt and prompt.text:
    send(prompt.text)
