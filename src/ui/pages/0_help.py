"""
Help & Docs

In-app reference documentation for the DOE Toolkit algorithms, rendered
from the Markdown sources in docs/algorithms/ so the docs travel with the
installed app (including the offline desktop build).
"""
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st

from src.ui.utils.docs import list_algo_docs, read_algo_doc
from src.ui.utils.state_management import initialize_session_state
from src.ui.components.sidebar import build_standard_sidebar

# Page configuration
st.set_page_config(
    page_title="Help & Docs",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize state
initialize_session_state()

# Standard sidebar
build_standard_sidebar()

st.title("📖 Help & Docs")
st.markdown("""
Welcome to the DOE Toolkit reference documentation. These pages describe the
algorithms behind each design type and analysis step: how they work, when to
choose them, and how results are computed.
""")

docs = list_algo_docs()

if not docs:
    st.warning("No documentation found. The docs/algorithms/ folder is missing.")
    st.stop()

st.markdown("### Select a topic")

col_sel, col_info = st.columns([1, 2])

with col_sel:
    topic = st.selectbox(
        "Documentation topic",
        options=[slug for slug, _ in docs],
        format_func=lambda slug: dict(docs)[slug],
        label_visibility="collapsed",
    )

with col_info:
    st.caption(f"**{dict(docs)[topic]}** — docs/algorithms/{topic}.md")

st.divider()

try:
    content = read_algo_doc(topic)
    st.markdown(content)
except KeyError as exc:
    st.error(str(exc))