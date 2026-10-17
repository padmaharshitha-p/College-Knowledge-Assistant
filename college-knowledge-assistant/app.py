from __future__ import annotations

import os
from typing import Any

import streamlit as st

from config.settings import settings
from frontend.components import (
    render_dashboard,
    render_document_library,
    render_landing_page,
    render_main_chat,
    render_settings,
)
from services.analytics_service import AnalyticsService
from services.document_service import DocumentService
from services.llm_service import OllamaService
from services.rag_service import RAGService

st.set_page_config(page_title="College Knowledge Assistant", page_icon="🎓", layout="wide")


@st.cache_resource
def get_services() -> dict[str, Any]:
    analytics = AnalyticsService(db_path=settings.sqlite_db_path)
    document_service = DocumentService(upload_dir=settings.uploads_dir)
    llm_service = OllamaService(base_url=settings.ollama_base_url, timeout=settings.ollama_timeout_seconds)
    rag_service = RAGService(document_service=document_service, llm_service=llm_service, vector_store_path=settings.chroma_dir)
    return {
        "analytics": analytics,
        "document_service": document_service,
        "llm_service": llm_service,
        "rag_service": rag_service,
    }


services = get_services()


def main() -> None:
    if "page" not in st.session_state:
        st.session_state.page = "Home"

    nav = st.sidebar.radio(
        "Navigation",
        ["Home", "Chat", "Documents", "Analytics", "Settings"],
        index=0,
    )
    st.session_state.page = nav

    if nav == "Home":
        render_landing_page()
    elif nav == "Chat":
        render_main_chat(services)
    elif nav == "Documents":
        render_document_library(services)
    elif nav == "Analytics":
        render_dashboard(services)
    elif nav == "Settings":
        render_settings(services)


if __name__ == "__main__":
    main()
