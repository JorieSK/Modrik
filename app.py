import os
from dotenv import load_dotenv
import streamlit as st

load_dotenv()
api_key = os.getenv("API_KEY")

st.title("first app Streamlit")

if "messages" not in st.session_state:
    st.session_state.messages = []

uploaded_file = st.file_uploader("Upload a file")
if uploaded_file is not None:
    st.session_state.messages.append({"type": "file", "name": uploaded_file.name})

for msg in st.session_state.messages:
    with st.chat_message("user"):
        if msg["type"] == "text":
            st.write(msg["content"])
        else:
            st.write(f"📄 Uploaded file: {msg['name']}")

prompt = st.chat_input("Type your message")
if prompt:
    st.session_state.messages.append({"type": "text", "content": prompt})
    st.rerun()