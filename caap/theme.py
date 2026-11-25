"""Custom CSS for the research workstation."""

from __future__ import annotations

import streamlit as st

CSS = """
<style>
  @import url('https://fonts.googleapis.com/css2?family=Source+Sans+3:wght@400;500;600;700&display=swap');

  html, body, [class*="css"] {
    font-family: "Source Sans 3", "Segoe UI", sans-serif;
  }

  .stApp {
    background:
      radial-gradient(980px 520px at -10% -20%, rgba(55, 118, 204, 0.28), transparent 58%),
      radial-gradient(760px 420px at 110% -10%, rgba(24, 74, 138, 0.22), transparent 52%),
      #071422;
    color: #E6EEF8;
  }

  section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0B1A30 0%, #081424 100%);
    border-right: 1px solid rgba(146, 184, 230, 0.14);
  }

  section[data-testid="stSidebar"] .block-container {
    padding-top: 1.4rem;
  }

  [data-testid="stSidebarNav"] {
    padding-top: 0.2rem;
  }

  [data-testid="stSidebarNav"] a {
    border-radius: 10px;
    margin: 2px 0;
  }

  header[data-testid="stHeader"] {
    background: rgba(7, 20, 34, 0.72);
    border-bottom: 1px solid rgba(146, 184, 230, 0.08);
  }

  footer, #MainMenu {
    visibility: hidden;
  }

  .block-container {
    padding-top: 1.6rem;
    padding-bottom: 3rem;
    max-width: 1180px;
  }

  .brand-block {
    padding: 0 0.2rem 0.8rem;
    border-bottom: 1px solid rgba(146, 184, 230, 0.12);
    margin-bottom: 0.8rem;
  }
  .brand-block .mark {
    color: #8EBCFF;
    letter-spacing: 0.16em;
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
  }
  .brand-block h2 {
    color: #F4F8FF;
    font-size: 1.35rem;
    font-weight: 650;
    margin: 0.15rem 0 0.2rem;
    letter-spacing: -0.03em;
  }
  .brand-block p {
    color: #9FB3CE;
    font-size: 0.86rem;
    line-height: 1.4;
    margin: 0;
  }

  .page-head {
    margin: 0.2rem 0 1.2rem;
  }
  .page-head .kicker {
    color: #8EBCFF;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    font-size: 0.75rem;
    font-weight: 700;
  }
  .page-head h1 {
    color: #F5F8FD;
    font-size: 2.15rem;
    line-height: 1.15;
    font-weight: 650;
    letter-spacing: -0.035em;
    margin: 0.25rem 0 0.45rem;
  }
  .page-head p {
    color: #B7C7DC;
    font-size: 1.05rem;
    line-height: 1.55;
    max-width: 760px;
    margin: 0;
  }

  div[data-testid="stMetric"] {
    background: linear-gradient(180deg, rgba(18, 44, 78, 0.92), rgba(10, 26, 48, 0.92));
    border: 1px solid rgba(146, 184, 230, 0.16);
    border-radius: 16px;
    padding: 0.85rem 1rem 0.65rem;
    box-shadow: 0 12px 28px rgba(0, 0, 0, 0.16);
  }
  div[data-testid="stMetric"] label {
    color: #9FB3CE !important;
  }

  .panel, .score-panel, .path-card, .note-card {
    background: linear-gradient(180deg, rgba(16, 38, 68, 0.88), rgba(9, 22, 42, 0.9));
    border: 1px solid rgba(146, 184, 230, 0.16);
    border-radius: 16px;
    padding: 1rem 1.1rem;
  }

  .score-panel {
    min-height: 168px;
  }
  .score-panel .score-kicker {
    color: #8EBCFF;
    font-size: 0.75rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    font-weight: 700;
  }
  .score-panel .score-value {
    color: #F7FBFF;
    font-size: 3rem;
    font-weight: 650;
    letter-spacing: -0.04em;
    line-height: 1.05;
    margin: 0.2rem 0;
  }
  .score-panel .score-label {
    color: #D5E4F7;
    font-size: 1rem;
  }
  .score-panel .score-sub {
    color: #9FB3CE;
    margin-top: 0.45rem;
    font-size: 0.92rem;
    line-height: 1.45;
  }
  .pill {
    display: inline-block;
    margin-top: 0.7rem;
    padding: 0.22rem 0.6rem;
    border-radius: 999px;
    border: 1px solid rgba(146, 184, 230, 0.28);
    color: #D7E6F8;
    font-size: 0.82rem;
    letter-spacing: 0.02em;
  }

  .path-card {
    min-height: 148px;
    margin-bottom: 0.4rem;
  }
  .path-card.active {
    border-color: rgba(94, 234, 212, 0.7);
    box-shadow: inset 0 0 0 1px rgba(94, 234, 212, 0.25);
  }
  .path-card h3 {
    color: #F4F8FF;
    font-size: 1.05rem;
    margin: 0.15rem 0 0.35rem;
  }
  .path-card p {
    color: #B7C7DC;
    margin: 0;
    line-height: 1.45;
    font-size: 0.92rem;
  }
  .path-card .state {
    color: #8EBCFF;
    font-size: 0.75rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    font-weight: 700;
  }
  .path-card.active .state {
    color: #5EEAD4;
  }

  .note-card {
    margin: 0.4rem 0 1rem;
  }
  .note-card strong {
    color: #F4F8FF;
  }
  .note-card p {
    color: #C5D4E8;
    margin: 0.25rem 0 0;
    line-height: 1.55;
  }

  .img-label {
    color: #D5E4F7;
    font-weight: 650;
    margin-bottom: 0.35rem;
  }

  div[data-testid="stImage"] {
    background: #050d18;
    border: 1px solid rgba(146, 184, 230, 0.16);
    border-radius: 14px;
    padding: 0.35rem;
  }

  div[data-testid="stDataFrame"] {
    border: 1px solid rgba(146, 184, 230, 0.14);
    border-radius: 14px;
    overflow: hidden;
  }

  .stButton > button, .stDownloadButton > button {
    border-radius: 10px;
    border: 1px solid rgba(142, 188, 255, 0.35);
    background: linear-gradient(180deg, #1A446F, #123255);
    color: #F4F8FF;
  }
  .stButton > button:hover, .stDownloadButton > button:hover {
    border-color: #8EBCFF;
    color: white;
  }

  div[data-testid="stAlert"] {
    border-radius: 12px;
  }
</style>
"""


def inject() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def page_header(kicker: str, title: str, lede: str) -> None:
    st.markdown(
        f"""
        <div class="page-head">
          <div class="kicker">{kicker}</div>
          <h1>{title}</h1>
          <p>{lede}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def note(title: str, body: str) -> None:
    st.markdown(
        f"""
        <div class="note-card">
          <strong>{title}</strong>
          <p>{body}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
