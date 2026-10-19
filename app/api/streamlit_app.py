# streamlit_app.py
import os

import requests
import streamlit as st

# Where the FastAPI server is running. Override with the API_URL environment variable.
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Document Q&A", page_icon="📄")
st.title("📄 Ask your document")


if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for i, chunk in enumerate(msg["sources"], 1):
                    st.markdown(f"**[{i}]** {chunk}")

question = st.chat_input("Ask a question about the document")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = requests.post(
                    f"{API_URL}/query",
                    json={"question": question},
                    timeout=120,
                )
                response.raise_for_status()
                data = response.json()
                answer, sources = data["answer"], data.get("sources", [])
            except requests.exceptions.RequestException as e:
                answer, sources = f"Could not get an answer from the API: {e}", []

        st.markdown(answer)
        if sources:
            with st.expander("Sources"):
                for i, chunk in enumerate(sources, 1):
                    st.markdown(f"**[{i}]** {chunk}")

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )