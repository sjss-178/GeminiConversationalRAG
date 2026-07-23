import streamlit as st
import os
import dotenv
import uuid
import os

os.environ["USER_AGENT"] = "My-RAG-App/1.0"

# Rest of your imports/code
# Required for Streamlit Cloud


from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage

from rag_methods import (
    load_doc_to_db,
    load_url_to_db,
    stream_llm_response,
    stream_llm_rag_response,
)

dotenv.load_dotenv()

# ----------------------------
# Gemini Models
# ----------------------------

MODELS = [
    "gemini-3-flash-preview"
]

# ----------------------------
# Streamlit Config
# ----------------------------

st.set_page_config(
    page_title="Gemini RAG Chat",
    page_icon="🤖",
    layout="wide",
)

st.title("📚 Gemini Conversational RAG")

# ----------------------------
# Session State
# ----------------------------

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "rag_sources" not in st.session_state:
    st.session_state.rag_sources = []

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello 👋 Upload documents or a website and ask me anything."
        }
    ]

if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

# ----------------------------
# Sidebar
# ----------------------------

with st.sidebar:

    st.header("🔑 Google Gemini")

    default_key = os.getenv("GOOGLE_API_KEY", "")

    

    st.divider()

    st.selectbox(
        "Model",
        MODELS,
        key="model",
    )

    st.toggle(
        "Use RAG",
        key="use_rag",
        value=st.session_state.vector_db is not None,
        disabled=st.session_state.vector_db is None,
    )

    if st.button("🗑 Clear Chat"):
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Chat cleared."
            }
        ]

    st.divider()

    st.header("📄 Upload Documents")

    st.file_uploader(
        "PDF / DOCX / TXT / MD",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True,
        key="rag_docs",
        on_change=load_doc_to_db,
    )

    st.divider()

    st.header("🌐 Website")

    st.text_input(
        "Enter URL",
        placeholder="https://example.com",
        key="rag_url",
        on_change=load_url_to_db,
    )

    st.divider()

    st.header("Indexed Sources")

    if len(st.session_state.rag_sources) == 0:
        st.info("No documents loaded.")
    else:
        for src in st.session_state.rag_sources:
            st.write("•", src)

# ----------------------------
# API Key Check
# ----------------------------

if default_key.strip() == "":
    st.warning("Please enter your Google Gemini API Key.")
    st.stop()

# ----------------------------
# Gemini LLM
# ----------------------------

llm = ChatGoogleGenerativeAI(
    model=st.session_state.model,
    google_api_key=default_key,
    temperature=0.2,
)

# ----------------------------
# Display Chat
# ----------------------------

for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ----------------------------
# Chat Input
# ----------------------------

prompt = st.chat_input("Ask me anything...")

if prompt:

    # Store user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt,
        }
    )

    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Convert Streamlit history to LangChain messages
    messages = []

    for m in st.session_state.messages:

        if m["role"] == "user":
            messages.append(
                HumanMessage(content=m["content"])
            )
        else:
            messages.append(
                AIMessage(content=m["content"])
            )

    # Generate assistant response
    with st.chat_message("assistant"):

        if (
            st.session_state.use_rag
            and st.session_state.vector_db is not None
        ):

            st.write_stream(
                stream_llm_rag_response(
                    llm,
                    messages,
                )
            )

        else:

            st.write_stream(
                stream_llm_response(
                    llm,
                    messages,
                )
            )

# ----------------------------
# Footer
# ----------------------------

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.caption("🚀 Gemini Conversational RAG")

with col2:
    st.caption("Built using LangChain + Chroma + Streamlit")