from __future__ import annotations

import os
from typing import Any

import streamlit as st

from config.settings import settings
from services.document_service import DocumentService


def _render_header(title: str, subtitle: str | None = None) -> None:
    st.subheader(title)
    if subtitle:
        st.caption(subtitle)


def render_landing_page() -> None:
    st.title("Your College Knowledge, One Intelligent Assistant")
    st.caption("Upload academic PDFs, ask grounded questions, and get answers backed by the documents you trust.")

    col_left, col_right = st.columns([1.4, 1])
    with col_left:
        st.markdown(
            """
            <div class="hero">
                <h3>Built for students, faculty, and academic support teams</h3>
                <p>College Knowledge Assistant combines retrieval-augmented generation with local document ingestion to help you find and explain relevant syllabus, regulation, and course information from uploaded academic sources.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Start Chatting", use_container_width=True):
            st.session_state.page = "Chat"
            st.rerun()
        if st.button("Upload Documents", use_container_width=True):
            st.session_state.page = "Documents"
            st.rerun()
    with col_right:
        st.markdown(
            """
            <div class="card">
                <h4>How it works</h4>
                <ol>
                    <li>Upload PDF documents</li>
                    <li>Chunk and embed the text</li>
                    <li>Retrieve relevant passages</li>
                    <li>Ground the answer in source evidence</li>
                </ol>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### Core features")
    feature_cols = st.columns(4)
    features = [
        ("📚", "Academic PDFs", "Syllabi, regulations, notes, and manuals."),
        ("🧠", "Local RAG", "Relevant passages are retrieved before answering."),
        ("📎", "Citations", "Sources and page context are preserved in answers."),
        ("🛡️", "Grounded answers", "Missing evidence produces a clear abstention."),
    ]
    for col, item in zip(feature_cols, features):
        with col:
            st.markdown(f"<div class='card'><h4>{item[0]} {item[1]}</h4><p>{item[2]}</p></div>", unsafe_allow_html=True)

    st.markdown("### Supported document categories")
    st.write("Course syllabus • Subject notes • Academic regulations • Examination guidelines • College handbooks • Department documents • Laboratory manuals • FAQs")

    st.markdown("### Footer")
    st.caption("Prototype for academic document Q&A using local LLMs and document-grounded retrieval.")


def render_main_chat(services: dict[str, Any]) -> None:
    st.title("Academic chat workspace")
    col_model, col_filter = st.columns([1, 2])
    with col_model:
        model_names = services["llm_service"].get_available_models()
        default_model = settings.default_ollama_model if settings.default_ollama_model in model_names else (model_names[0] if model_names else settings.default_ollama_model)
        selected_model = st.selectbox("Model", model_names or [settings.default_ollama_model], index=(model_names.index(default_model) if default_model in model_names else 0))
        st.caption(f"Selected model: {selected_model}")
    with col_filter:
        subject_filter = st.text_input("Subject filter (optional)", value="")
        top_k = st.slider("Top-k retrieval", min_value=1, max_value=8, value=settings.top_k)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_question = st.chat_input("Ask a question about the uploaded academic documents...")

    if user_question:
        st.session_state.chat_history.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        try:
            response = services["rag_service"].answer_question(
                question=user_question,
                model_name=selected_model,
                subject_filter=subject_filter.strip() or None,
                top_k=top_k,
            )
        except Exception as exc:
            response = {
                "answer": "I couldn't find sufficient information to answer this question in the uploaded college documents.",
                "sources": [],
                "follow_up_questions": [
                    "Please upload a relevant PDF document.",
                    "Confirm the selected model is installed and available.",
                ],
                "grounded": False,
                "error": str(exc),
            }

        with st.chat_message("assistant"):
            st.markdown(response.get("answer", "I couldn't find sufficient information to answer this question in the uploaded college documents."))
            if response.get("sources"):
                st.markdown("#### Sources")
                for src in response["sources"]:
                    st.markdown(f"- {src.get('filename')} — Page {src.get('page_number')}" if src.get('page_number') else f"- {src.get('filename')}")
        st.session_state.chat_history.append({"role": "assistant", "content": response.get("answer", "I couldn't find sufficient information to answer this question in the uploaded college documents.")})

    if st.button("Clear conversation"):
        st.session_state.chat_history = []


def render_document_library(services: dict[str, Any]) -> None:
    st.title("Document library")
    st.caption("Upload academic PDFs to index and retrieve from the local knowledge base.")

    uploaded = st.file_uploader("Upload PDF documents", type=["pdf"], accept_multiple_files=True)
    if uploaded:
        for file in uploaded:
            try:
                services["document_service"].index_pdf_upload(file)
                st.success(f"Indexed document: {file.name}")
            except Exception as exc:  # pragma: no cover - UI fallback
                st.error(str(exc))

    documents = services["document_service"].list_documents()
    if not documents:
        st.info("No documents uploaded yet. Add one or more PDFs to start indexing.")
        return

    st.dataframe(documents, use_container_width=True)

    for doc in documents:
        col1, col2, col3 = st.columns([3, 2, 1])
        with col1:
            st.write(doc.get("filename", "Unknown document"))
        with col2:
            st.write(doc.get("status", "unknown"))
        with col3:
            if st.button("Delete", key=f"delete-{doc['id']}"):
                services["document_service"].delete_document(doc["id"])
                st.rerun()


def render_dashboard(services: dict[str, Any]) -> None:
    st.title("Analytics dashboard")
    analytics = services["analytics"].read_summary()
    if not analytics:
        st.info("No usage data is available yet. Upload a document and ask a question to populate the dashboard.")
        return
    cols = st.columns(4)
    for i, metric in enumerate([("Indexed documents", analytics.get("document_count", 0)), ("Indexed chunks", analytics.get("chunk_count", 0)), ("Questions asked", analytics.get("question_count", 0)), ("Recent activity", analytics.get("last_activity", "N/A"))]):
        with cols[i]:
            st.markdown(f"<div class='kpi'><h4>{metric[0]}</h4><h2>{metric[1]}</h2></div>", unsafe_allow_html=True)


def render_settings(services: dict[str, Any]) -> None:
    st.title("Settings")
    st.caption("Local model and retrieval configuration")
    model_names = services["llm_service"].get_available_models()
    if model_names:
        st.selectbox("Default model", model_names, index=0, key="default_model")
    else:
        st.warning("No Ollama models were detected. Start Ollama and run 'ollama list' to inspect local availability.")
    st.slider("Chunk size", min_value=300, max_value=1600, value=settings.chunk_size, key="chunk_size")
    st.slider("Chunk overlap", min_value=40, max_value=300, value=settings.chunk_overlap, key="chunk_overlap")
    st.slider("Top-k retrieval", min_value=1, max_value=10, value=settings.top_k, key="top_k")
    threshold = st.slider("Retrieval similarity threshold", min_value=0.0, max_value=1.0, value=settings.retrieval_similarity_threshold, step=0.05, key="retrieval_similarity_threshold")
    settings.retrieval_similarity_threshold = float(threshold)
    st.checkbox("Theme preference", value=False, key="theme_mode")

    show_debug = st.checkbox("Developer retrieval diagnostics", value=False, key="dev_retrieval_debug")
    if show_debug:
        debug = services["rag_service"].get_last_retrieval_debug()
        if debug:
            st.json(debug)
        else:
            st.info("No retrieval has been run yet. Ask a question in Chat to generate debug output.")
