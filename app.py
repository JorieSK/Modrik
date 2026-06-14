import os
from dotenv import load_dotenv
import streamlit as st
from ai_service import stream_response
from file_service import extract_text

load_dotenv()

st.title("first app Streamlit")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "file_context" not in st.session_state:
    st.session_state.file_context = ""

uploaded_file = st.file_uploader("Upload a file")
if uploaded_file is not None:
    st.session_state.file_context = extract_text(uploaded_file)

for msg in st.session_state.messages:
    role = msg.get("role", "user")
    with st.chat_message(role):
        if msg["type"] == "text":
            st.write(msg["content"])
        elif msg["type"] == "file":
            st.write(f"📄 Uploaded file: {msg['name']}")

prompt = st.chat_input("Type your message")
if prompt:
    st.session_state.messages.append({"role": "user", "type": "text", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        response = st.write_stream(
            stream_response(prompt, context=st.session_state.file_context)
        )

    st.session_state.messages.append({"role": "assistant", "type": "text", "content": response})
