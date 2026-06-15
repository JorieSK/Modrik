import os
from dotenv import load_dotenv
import streamlit as st
from ai_service import stream_response
from file_service import extract_text
from vector_store import load_from_disk, retrieve_with_scores, is_loaded

load_dotenv()

st.set_page_config(
    page_title="مدرك",
    page_icon="✦",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    #MainMenu, footer, header { visibility: hidden; }
    [data-testid="stSidebar"] { display: none; }
    .block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 780px; }
    body, html { background: #ffffff; }

    /* ── Header ── */
    .top-header { text-align: center; padding: 0.8rem 0 0.6rem; }
    .top-header h1 { font-size: 2.2rem; font-weight: 700; color: #1A9AA8; margin: 0.3rem 0 0.2rem; }
    .top-header p  { font-size: 0.88rem; color: #7DCDD4; margin: 0; }

    /* ── Stats strip ── */
    .stats-strip {
        display: flex;
        justify-content: center;
        gap: 1.2rem;
        margin: 1.2rem 0;
    }
    .stat-card {
        flex: 1;
        background: #F4FCFD;
        border: 1.5px solid #B2E8EC;
        border-radius: 12px;
        padding: 0.9rem 0.5rem;
        text-align: center;
    }
    .stat-number { font-size: 1.5rem; font-weight: 700; color: #1A9AA8; line-height: 1; }
    .stat-label  { font-size: 0.7rem; color: #7DCDD4; margin-top: 0.3rem; letter-spacing: 0.04em; text-transform: uppercase; }

    /* ── Feature pills ── */
    .features {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        justify-content: center;
        margin: 0.8rem 0 1rem;
    }
    .feat-pill {
        background: #E6F9FA;
        border: 1.5px solid #4ECDD4;
        border-radius: 20px;
        padding: 0.3rem 0.85rem;
        font-size: 0.78rem;
        color: #1A9AA8;
        white-space: nowrap;
    }

    /* ── Divider ── */
    .divider { border: none; border-top: 1.5px solid #D0F0F3; margin: 1rem 0; }

    /* ── Upload section label ── */
    .section-label {
        font-size: 0.7rem;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #7DCDD4;
        margin-bottom: 0.4rem;
        font-weight: 600;
    }

    /* ── File tag ── */
    .file-tag {
        background: #E6F9FA;
        border: 1.5px solid #4ECDD4;
        border-radius: 8px;
        padding: 0.35rem 0.8rem;
        font-size: 0.8rem;
        color: #1A9AA8;
        margin-top: 0.5rem;
        display: inline-block;
    }

    /* ── Suggested prompts ── */
    .prompts-label {
        font-size: 0.7rem;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #7DCDD4;
        margin: 1rem 0 0.5rem;
        font-weight: 600;
    }
    .prompt-chips {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin-bottom: 0.5rem;
    }
    .prompt-chip {
        background: white;
        border: 1.5px solid #B2E8EC;
        border-radius: 20px;
        padding: 0.35rem 0.9rem;
        font-size: 0.8rem;
        color: #1A9AA8;
        cursor: default;
        transition: all 0.15s;
    }
    .prompt-chip:hover { background: #E6F9FA; border-color: #4ECDD4; }

    /* ── Buttons ── */
    .stButton > button {
        background: white !important;
        border: 1.5px solid #4ECDD4 !important;
        color: #1A9AA8 !important;
        border-radius: 8px !important;
        font-size: 0.78rem !important;
        padding: 0.3rem 0.8rem !important;
        transition: all 0.15s;
    }
    .stButton > button:hover {
        background: #E6F9FA !important;
        border-color: #1A9AA8 !important;
    }

    /* ── File uploader ── */
    [data-testid="stFileUploader"] section {
        border: 1.5px dashed #4ECDD4 !important;
        border-radius: 10px !important;
        background: #F4FCFD !important;
        padding: 0.6rem !important;
    }

    /* ── Chat input ── */
    [data-testid="stChatInput"] textarea {
        border: 1.5px solid #4ECDD4 !important;
        border-radius: 10px !important;
        background: #F4FCFD !important;
        font-size: 0.9rem !important;
    }
    [data-testid="stChatInput"] textarea:focus {
        border-color: #1A9AA8 !important;
        box-shadow: 0 0 0 3px rgba(78,205,212,0.2) !important;
    }

    /* ── Empty state ── */
    .empty { text-align: center; padding: 3rem 1rem; }
    .empty h2 { font-size: 1rem; color: #4ECDD4; margin: 0.5rem 0 0.3rem; font-weight: 500; }
    .empty p  { font-size: 0.82rem; color: #A8E0E4; }

    /* ── Footer ── */
    .footer {
        text-align: center;
        padding: 1.5rem 0 0.5rem;
        font-size: 0.72rem;
        color: #B2E8EC;
        border-top: 1.5px solid #D0F0F3;
        margin-top: 1rem;
    }
    .footer span { color: #4ECDD4; }
</style>
""", unsafe_allow_html=True)

# ── Load vector index once per session ────────────────────────
if "rag_ready" not in st.session_state:
    st.session_state.rag_ready = load_from_disk()

# ── Session state ──────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = ""
if "file_text" not in st.session_state:
    st.session_state.file_text = ""
if "file_name" not in st.session_state:
    st.session_state.file_name = ""

# ── Logo + title ───────────────────────────────────────────────
col1, col2, col3 = st.columns([2, 1, 2])
with col2:
    st.image("logo.jpeg", use_container_width=True)

st.markdown("""
<div class="top-header">
    <h1>مدرك</h1>
    <p>Your AI-powered document assistant — ask anything, in any language</p>
</div>
""", unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ── RAG status + clear ─────────────────────────────────────────
status_col, btn_col = st.columns([5, 1])
with status_col:
    if st.session_state.rag_ready:
        st.markdown('<div class="file-tag">⚖️ نظام العمل السعودي — محمّل وجاهز</div>', unsafe_allow_html=True)
    else:
        st.warning("لم يتم تحميل قاعدة البيانات. شغّل: `python3 ingest_labor_law.py` أولاً.", icon="⚠️")
with btn_col:
    st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

# ── Case file upload ────────────────────────────────────────────
st.markdown('<div class="section-label" style="margin-top:0.8rem;">Upload Case File (optional)</div>', unsafe_allow_html=True)
up_col, remove_col = st.columns([5, 1])
with up_col:
    uploaded_file = st.file_uploader("case file", type=["txt", "pdf"], label_visibility="collapsed")
    if uploaded_file:
        if uploaded_file.name != st.session_state.file_name:
            with st.spinner("Reading file…"):
                st.session_state.file_text = extract_text(uploaded_file)
                st.session_state.file_name = uploaded_file.name
        st.markdown(f'<div class="file-tag">📎 {uploaded_file.name}</div>', unsafe_allow_html=True)
    else:
        st.session_state.file_text = ""
        st.session_state.file_name = ""

# ── Suggested prompts ──────────────────────────────────────────
if not st.session_state.messages:
    st.markdown('<div class="prompts-label">Try asking</div>', unsafe_allow_html=True)
    suggestions = [
        "ما هي حقوق العامل عند إنهاء العقد؟",
        "كم عدد ساعات العمل الأسبوعية؟",
        "ما شروط إجازة الأمومة؟",
        "متى يستحق العامل مكافأة نهاية الخدمة؟",
    ]
    cols = st.columns(2)
    for i, s in enumerate(suggestions):
        with cols[i % 2]:
            if st.button(s, key=f"sug_{i}"):
                st.session_state.pending_prompt = s
                st.rerun()

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ── Helper functions ────────────────────────────────────────────
def show_sources(scored_chunks: list[tuple[str, float]], file_text: str):
    parts = []
    if scored_chunks:
        parts.append(f"⚖️ {len(scored_chunks)} مواد من نظام العمل")
    if file_text:
        parts.append("📎 ملف العقد")
    if not parts:
        return
    with st.expander(f"📚 المصادر المستخدمة — {' · '.join(parts)}"):
        if file_text:
            st.markdown("**📎 ملف العقد المرفوع**")
            st.markdown(f"> {file_text[:1500].replace(chr(10), '  \n> ')}{'…' if len(file_text) > 1500 else ''}")
            if scored_chunks:
                st.divider()
        for i, (chunk, score) in enumerate(scored_chunks, 1):
            pct = int(score * 100)
            color = "#1A9AA8" if pct >= 70 else "#F39C12" if pct >= 50 else "#999"
            st.markdown(
                f"**⚖️ المادة {i}** &nbsp; "
                f'<span style="font-size:0.75rem;color:{color};">تطابق {pct}%</span>',
                unsafe_allow_html=True,
            )
            st.markdown(f"> {chunk.replace(chr(10), '  \n> ')}")
            if i < len(scored_chunks):
                st.divider()

def send(prompt: str):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    scored_chunks = retrieve_with_scores(prompt, k=5) if is_loaded() else []
    file_text = st.session_state.file_text

    context_parts = []
    if file_text:
        context_parts.append(f"=== ملف القضية ===\n{file_text}")
    if scored_chunks:
        law_text = "\n\n---\n\n".join(chunk for chunk, _ in scored_chunks)
        context_parts.append(f"=== مواد نظام العمل ذات الصلة ===\n{law_text}")
    context = "\n\n".join(context_parts)

    with st.chat_message("assistant"):
        response = st.write_stream(stream_response(prompt, context=context))

    show_sources(scored_chunks, file_text)
    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
        "scored_chunks": scored_chunks,
        "file_text": file_text,
    })

# ── Chat ───────────────────────────────────────────────────────
if not st.session_state.messages:
    st.markdown("""
    <div class="empty">
        <h2>ابدأ المحادثة</h2>
        <p>ارفع عقدك أعلاه ثم اسألني عن وضعك القانوني.</p>
    </div>
    """, unsafe_allow_html=True)
else:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
        if msg["role"] == "assistant":
            show_sources(msg.get("scored_chunks", []), msg.get("file_text", ""))

if st.session_state.pending_prompt:
    p = st.session_state.pending_prompt
    st.session_state.pending_prompt = ""
    send(p)

prompt = st.chat_input("اسألني عن وضعك في العقد…")
if prompt:
    send(prompt)

# ── Footer (decorative) ────────────────────────────────────────
st.markdown("""
<div class="footer">
    Built with ❤ · Powered by <span>GPT-4o mini</span> · مدرك © 2025
</div>
""", unsafe_allow_html=True)
