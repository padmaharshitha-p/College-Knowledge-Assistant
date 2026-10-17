from __future__ import annotations

import streamlit as st


def apply_global_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg: #f4f7fb;
            --navy: #0f172a;
            --royal: #1d4ed8;
            --violet: #7c3aed;
            --muted: #475569;
            --card: #ffffff;
            --border: rgba(15, 23, 42, 0.12);
        }
        .stApp {
            background: linear-gradient(180deg, #eef4ff 0%, #f8fafc 100%);
            color: var(--navy);
        }
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 3rem;
        }
        div[data-testid="stSidebar"] {
            background: #0f172a;
            color: white;
        }
        .card {
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 18px;
            padding: 1rem 1.1rem;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.05);
        }
        .hero {
            background: linear-gradient(135deg, rgba(29,78,216,0.12), rgba(124,58,237,0.12));
            border: 1px solid rgba(29,78,216,0.14);
            border-radius: 26px;
            padding: 2rem;
        }
        .kpi {
            background: white;
            border: 1px solid var(--border);
            border-radius: 18px;
            padding: 1rem;
        }
        .source-box {
            background: #f8fafc;
            border-left: 4px solid #7c3aed;
            border-radius: 10px;
            padding: 0.7rem 0.9rem;
            margin-top: 0.6rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
