import os
import dotenv
from time import time

import streamlit as st

from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    WebBaseLoader,
)

from langchain_community.document_loaders.text import TextLoader

from langchain_community.vectorstores import Chroma

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_google_genai import GoogleGenerativeAIEmbeddings

from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)

from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.history_aware_retriever import (
    create_history_aware_retriever
)

dotenv.load_dotenv()

os.environ["USER_AGENT"] = "gemini-rag"

DB_DOCS_LIMIT = 10


####################################################
# NORMAL CHAT
####################################################

def stream_llm_response(llm, messages):

    response = ""

    for chunk in llm.stream(messages):

        response += chunk.content

        yield chunk.content

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response,
        }
    )


####################################################
# DOCUMENT LOADING
####################################################

def load_doc_to_db():

    if "rag_docs" not in st.session_state:
        return

    docs = []

    os.makedirs("source_files", exist_ok=True)

    for uploaded_file in st.session_state.rag_docs:

        if uploaded_file.name in st.session_state.rag_sources:
            continue

        if len(st.session_state.rag_sources) >= DB_DOCS_LIMIT:
            st.error("Maximum document limit reached.")
            return

        path = os.path.join(
            "source_files",
            uploaded_file.name,
        )

        with open(path, "wb") as f:
            f.write(uploaded_file.read())

        try:

            if uploaded_file.name.endswith(".pdf"):
                loader = PyPDFLoader(path)

            elif uploaded_file.name.endswith(".docx"):
                loader = Docx2txtLoader(path)

            elif uploaded_file.name.endswith(".txt"):

                loader = TextLoader(path)

            elif uploaded_file.name.endswith(".md"):

                loader = TextLoader(path)

            else:
                st.warning(
                    f"{uploaded_file.name} not supported."
                )
                continue

            docs.extend(loader.load())

            st.session_state.rag_sources.append(
                uploaded_file.name
            )

        finally:

            if os.path.exists(path):
                os.remove(path)

    if docs:

        split_and_store_docs(docs)

        st.success("Documents indexed successfully.")


####################################################
# URL LOADING
####################################################

def load_url_to_db():

    if (
        "rag_url" not in st.session_state
        or st.session_state.rag_url == ""
    ):
        return

    url = st.session_state.rag_url

    if url in st.session_state.rag_sources:
        return

    loader = WebBaseLoader(url)

    docs = loader.load()

    split_and_store_docs(docs)

    st.session_state.rag_sources.append(url)

    st.success("Website indexed.")


####################################################
# VECTOR DATABASE
####################################################

def initialize_vector_db(chunks):

    embeddings = GoogleGenerativeAIEmbeddings(

        model="models/gemini-embedding-001",

        google_api_key=st.session_state.google_api_key,

    )

    vector_db = Chroma.from_documents(

        documents=chunks,

        embedding=embeddings,

        collection_name=f"rag_{time()}_{st.session_state.session_id}",

    )

    client = vector_db._client

    collections = sorted(
        [c.name for c in client.list_collections()]
    )

    while len(collections) > 20:

        client.delete_collection(collections[0])

        collections.pop(0)

    return vector_db


####################################################
# SPLITTING
####################################################

def split_and_store_docs(docs):

    splitter = RecursiveCharacterTextSplitter(

        chunk_size=1200,

        chunk_overlap=250,

    )

    chunks = splitter.split_documents(docs)

    if st.session_state.vector_db is None:

        st.session_state.vector_db = initialize_vector_db(
            chunks
        )

    else:

        st.session_state.vector_db.add_documents(chunks)

####################################################
# HISTORY AWARE RETRIEVER
####################################################

def get_context_retriever_chain(vector_db, llm):

    retriever = vector_db.as_retriever(
        search_kwargs={"k": 4}
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            MessagesPlaceholder("messages"),

            (
                "user",
                "{input}",
            ),

            (
                "user",
                """
Given the conversation above, generate a standalone
search query that best represents the user's latest
question.

Only return the search query.
""",
            ),
        ]
    )

    return create_history_aware_retriever(
        llm,
        retriever,
        prompt,
    )


####################################################
# CONVERSATIONAL RAG CHAIN
####################################################

def get_conversational_rag_chain(llm):

    retriever_chain = get_context_retriever_chain(
        st.session_state.vector_db,
        llm,
    )

    prompt = ChatPromptTemplate.from_messages(
        [

            (
                "system",
                """
You are a helpful AI assistant.

Use the retrieved context below to answer
the user's question.

If the answer is not contained in the context,
say you don't know instead of making up an answer.

------------------------
{context}
------------------------
""",
            ),

            MessagesPlaceholder("messages"),

            (
                "user",
                "{input}",
            ),
        ]
    )

    document_chain = create_stuff_documents_chain(
        llm,
        prompt,
    )

    return create_retrieval_chain(
        retriever_chain,
        document_chain,
    )


####################################################
# STREAMING RAG RESPONSE
####################################################

def stream_llm_rag_response(llm, messages):

    rag_chain = get_conversational_rag_chain(
        llm
    )

    response = ""

    for chunk in rag_chain.pick("answer").stream(
        {
            "messages": messages[:-1],
            "input": messages[-1].content,
        }
    ):

        response += chunk

        yield chunk

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response,
        }
    )